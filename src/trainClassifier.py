"""
Fine-tune ModernBERT-large as a multi-label classifier for the labels in
materials/codebook.csv.
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from buildClassifierData import load_labels, load_rows
from generateResponses import ROOT_DIR, write_json


BASE_MODEL = "answerdotai/ModernBERT-large"
MAX_LENGTH = 256  # longest response is 104 words
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
EPOCHS = 4
LEARNING_RATES = [3e-5, 5e-5, 8e-5]
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
SEED = 1
ARTIFACTS_DIR = ROOT_DIR / "models" / "classifier"
# The shipped model, trained on FINAL_TRAIN_SPLITS.
MODEL_DIR = ARTIFACTS_DIR / "retrained" / "model"
# The sweep's best checkpoint, trained on TRAIN_SPLITS and tested on the rest.
DEV_MODEL_DIR = ARTIFACTS_DIR / "development" / "model"
TRAINING_LOG_PATH = ARTIFACTS_DIR / "training_log.jsonl"
SELECTION_PATH = ARTIFACTS_DIR / "selection.json"
# Written by learningCurve.py and read by evaluateClassifier.py.
LEARNING_CURVE_PATH = ROOT_DIR / "results" / "content_coding" / "learning_curve.jsonl"
TRAIN_SPLITS = {"train"}
FINAL_TRAIN_SPLITS = {"train", "test"}


def label_targets(
    rows: list[dict[str, Any]],
    label_keys: list[str],
    field: str = "annotated_label_keys",
) -> torch.Tensor:
    targets = [
        [float(key in keys) for key in label_keys]
        for keys in (set(row[field]) for row in rows)
    ]
    return torch.tensor(targets).reshape(len(rows), len(label_keys))


def predict_probabilities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    batch_size: int = EVAL_BATCH_SIZE,
) -> torch.Tensor:
    """Return label probabilities on the CPU, shaped [len(texts), num_labels]."""
    model.eval()
    batches: list[torch.Tensor] = []

    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).logits
            batches.append(torch.sigmoid(logits.float()).cpu())

    return torch.cat(batches)


def macro_ap(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Mean average precision over the labels that have at least one positive
    and one negative.
    """
    scores: list[float] = []

    for label in range(targets.shape[1]):
        positives = int(targets[:, label].sum())

        if 0 < positives < len(targets):
            scores.append(
                average_precision_score(
                    targets[:, label].numpy(),
                    probabilities[:, label].numpy(),
                )
            )

    return sum(scores) / len(scores)


def calibration_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Return the loss and the macro AP on the calibration rows."""
    probabilities = predict_probabilities(model, tokenizer, texts)
    loss = F.binary_cross_entropy(
        probabilities.clamp(1e-6, 1 - 1e-6),
        targets,
    ).item()
    return loss, macro_ap(probabilities, targets)


def train_model(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    targets: torch.Tensor,
    label_keys: list[str],
    learning_rate: float,
    schedule_epochs: int,
    epochs_to_run: int,
    seed: int,
    on_epoch: Callable[[PreTrainedModel, int, float, float], None],
) -> None:
    """
    Fine-tune a fresh BASE_MODEL on texts with binary cross-entropy.

    The learning-rate schedule spans schedule_epochs epochs, so stopping
    after epochs_to_run epochs reproduces those epochs of a full run. seed
    fixes the head initialisation and the batch order. After each epoch,
    on_epoch(model, epoch, train_loss, started) is called, where started is
    the epoch's time.perf_counter() start.
    """
    encodings = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
    features = [
        {"input_ids": input_ids, "attention_mask": attention_mask}
        for input_ids, attention_mask in zip(
            encodings["input_ids"],
            encodings["attention_mask"],
        )
    ]

    def collate(
        indices: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = tokenizer.pad(
            [features[index] for index in indices],
            return_tensors="pt",
        )
        return batch, targets[indices]

    random.seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(
        range(len(texts)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_keys),
        id2label=dict(enumerate(label_keys)),
        label2id={key: index for index, key in enumerate(label_keys)},
        problem_type="multi_label_classification",
        classifier_pooling="mean",
        attn_implementation="sdpa",
    ).to("cuda")

    # Norms and biases are not decayed.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.ndim >= 2
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.ndim < 2
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
    )
    total_steps = schedule_epochs * len(loader)
    warmup = math.ceil(WARMUP_FRACTION * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / warmup
            if step < warmup
            else max(0.0, (total_steps - step) / (total_steps - warmup))
        ),
    )

    for epoch in range(1, epochs_to_run + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device="cuda")

        for batch, batch_targets in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                ).logits

            # The loss is computed outside autocast, in float32.
            loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                batch_targets.to("cuda"),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sum += loss.detach()

        on_epoch(model, epoch, loss_sum.item() / len(loader), started)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "trainClassifier.py needs a CUDA GPU; "
            "torch.cuda.is_available() is False."
        )

    label_keys = [label.machine_key for label in load_labels()]
    train_rows = load_rows(TRAIN_SPLITS)
    calibration_rows = load_rows({"calibration"})
    final_rows = load_rows(FINAL_TRAIN_SPLITS)
    train_targets = label_targets(train_rows, label_keys)
    calibration_targets = label_targets(calibration_rows, label_keys)
    final_targets = label_targets(final_rows, label_keys)
    train_texts = [row["text"] for row in train_rows]
    calibration_texts = [row["text"] for row in calibration_rows]
    final_texts = [row["text"] for row in final_rows]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    token_counts = [
        len(input_ids)
        for input_ids in tokenizer(final_texts + calibration_texts)["input_ids"]
    ]
    truncated = sum(count > MAX_LENGTH for count in token_counts)
    print(
        f"Longest response: {max(token_counts)} tokens; "
        f"{truncated} truncated at {MAX_LENGTH}"
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    selection: dict[str, Any] = {}

    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as log_file:

        def record_epoch(
            phase: str,
            learning_rate: float,
            model: PreTrainedModel,
            epoch: int,
            train_loss: float,
            started: float,
        ) -> tuple[float, float]:
            """Score, log and print one epoch; return its calibration scores."""
            calibration_loss, calibration_macro_ap = calibration_scores(
                model,
                tokenizer,
                calibration_texts,
                calibration_targets,
            )
            seconds = time.perf_counter() - started
            log_file.write(
                json.dumps(
                    {
                        "phase": phase,
                        "learning_rate": learning_rate,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "calibration_loss": calibration_loss,
                        "calibration_macro_ap": calibration_macro_ap,
                        "seconds": seconds,
                    }
                )
                + "\n"
            )
            log_file.flush()
            print(
                f"{phase} lr={learning_rate:g} epoch {epoch}/{EPOCHS}: "
                f"train loss {train_loss:.4f}, "
                f"calibration loss {calibration_loss:.4f}, "
                f"macro AP {calibration_macro_ap:.4f} ({seconds:.0f} s)",
                flush=True,
            )
            return calibration_loss, calibration_macro_ap

        for learning_rate in LEARNING_RATES:

            def on_sweep_epoch(
                model: PreTrainedModel,
                epoch: int,
                train_loss: float,
                started: float,
                learning_rate: float = learning_rate,
            ) -> None:
                calibration_loss, calibration_macro_ap = record_epoch(
                    "sweep",
                    learning_rate,
                    model,
                    epoch,
                    train_loss,
                    started,
                )
                best_ap = selection.get("calibration_macro_ap", -1.0)

                if calibration_macro_ap > best_ap:
                    model.save_pretrained(DEV_MODEL_DIR)
                    tokenizer.save_pretrained(DEV_MODEL_DIR)
                    selection.update(
                        {
                            "base_model": BASE_MODEL,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "calibration_macro_ap": calibration_macro_ap,
                            "calibration_loss": calibration_loss,
                            "max_length": MAX_LENGTH,
                            "batch_size": BATCH_SIZE,
                            "seed": SEED,
                            "train_splits": sorted(TRAIN_SPLITS),
                        }
                    )
                    write_json(SELECTION_PATH, selection)

            train_model(
                tokenizer,
                train_texts,
                train_targets,
                label_keys,
                learning_rate,
                EPOCHS,
                EPOCHS,
                SEED,
                on_sweep_epoch,
            )

        print(
            f"Best: lr={selection['learning_rate']:g}, "
            f"epoch {selection['epoch']}, calibration macro AP "
            f"{selection['calibration_macro_ap']:.4f}. "
            f"Dev model: {DEV_MODEL_DIR}",
            flush=True,
        )

        def on_final_epoch(
            model: PreTrainedModel,
            epoch: int,
            train_loss: float,
            started: float,
        ) -> None:
            calibration_loss, calibration_macro_ap = record_epoch(
                "final",
                selection["learning_rate"],
                model,
                epoch,
                train_loss,
                started,
            )

            if epoch == selection["epoch"]:
                model.save_pretrained(MODEL_DIR)
                tokenizer.save_pretrained(MODEL_DIR)
                selection.update(
                    {
                        "final_train_splits": sorted(FINAL_TRAIN_SPLITS),
                        "final_calibration_macro_ap": calibration_macro_ap,
                        "final_calibration_loss": calibration_loss,
                    }
                )
                write_json(SELECTION_PATH, selection)

        train_model(
            tokenizer,
            final_texts,
            final_targets,
            label_keys,
            selection["learning_rate"],
            EPOCHS,
            selection["epoch"],
            SEED,
            on_final_epoch,
        )

    print(
        f"Shipped model (train + test dilemmas): calibration macro AP "
        f"{selection['final_calibration_macro_ap']:.4f}. Model: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
