"""
Measure how the classifier's test scores grow with the number of labelled
training answers, next to a word-count baseline trained on the same answers.
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from buildClassifierData import load_labels, load_rows
from evaluateClassifier import (
    EVALUATED_SPLITS,
    Predictions,
    all_label_metrics,
    macro,
    row_scores,
    tune_thresholds,
)
from trainClassifier import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_CURVE_PATH,
    SELECTION_PATH,
    TRAIN_SPLITS,
    label_targets,
    macro_ap,
    predict_probabilities,
    train_model,
)


# Training-set sizes below the full train split; the full split is run too.
TRAIN_SIZES = (250, 500, 1000, 2000, 5000, 10000)
# Each size below the full split is drawn once per seed
SEEDS = (1, 2, 3)
# Small sets train for more epochs, so every run takes at least this many
# optimizer steps.
MIN_TRAIN_STEPS = 1000
# Calibration checkpoints per run, at evenly spaced epochs.
EVALUATIONS_PER_RUN = 4
# Inverse regularisation strengths tried for the word-count baseline.
BASELINE_C_VALUES = (1.0, 4.0, 16.0)


def score_run(
    probabilities: dict[str, torch.Tensor],
    split_rows: dict[str, list[dict[str, Any]]],
    split_targets: dict[str, torch.Tensor],
    label_keys: list[str],
) -> dict[str, Any]:
    predictions = {
        split: Predictions(
            split_rows[split],
            probabilities[split],
            split_targets[split],
        )
        for split in EVALUATED_SPLITS
    }
    thresholds = tune_thresholds(predictions["calibration"])
    threshold_tensor = torch.tensor(thresholds)
    test_metrics = all_label_metrics(predictions["test"], thresholds)
    test_all_correct, test_pair_accuracy = row_scores(
        predictions["test"],
        threshold_tensor,
    )
    cross_generator_all_correct, _ = row_scores(
        predictions["cross_generator"],
        threshold_tensor,
    )
    return {
        "test_macro_f1": macro(test_metrics, "f1"),
        "test_macro_ap": macro(test_metrics, "ap"),
        "test_all_correct": test_all_correct,
        "test_pair_accuracy": test_pair_accuracy,
        "cross_generator_all_correct": cross_generator_all_correct,
        "label_f1": {
            key: entry["f1"] for key, entry in zip(label_keys, test_metrics)
        },
    }


def run_modernbert(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    targets: torch.Tensor,
    label_keys: list[str],
    learning_rate: float,
    seed: int,
    split_texts: dict[str, list[str]],
    split_targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    epochs = max(
        EPOCHS,
        math.ceil(MIN_TRAIN_STEPS / math.ceil(len(texts) / BATCH_SIZE)),
    )
    evaluated = sorted(
        {
            math.ceil(epochs * checkpoint / EVALUATIONS_PER_RUN)
            for checkpoint in range(1, EVALUATIONS_PER_RUN + 1)
        }
    )
    best: dict[str, Any] = {
        "epochs": epochs,
        "evaluated_epochs": evaluated,
        "calibration_macro_ap": -1.0,
    }

    def on_epoch(
        model: PreTrainedModel,
        epoch: int,
        train_loss: float,
        started: float,
    ) -> None:
        if not math.isfinite(train_loss):
            raise SystemExit(
                f"modernbert n={len(texts)} seed {seed}: train loss "
                f"{train_loss} after epoch {epoch}."
            )

        if epoch not in evaluated:
            return

        calibration = predict_probabilities(
            model,
            tokenizer,
            split_texts["calibration"],
        )
        calibration_macro_ap = macro_ap(
            calibration,
            split_targets["calibration"],
        )

        if calibration_macro_ap > best["calibration_macro_ap"]:
            best.update(
                {
                    "best_epoch": epoch,
                    "calibration_macro_ap": calibration_macro_ap,
                    "probabilities": {
                        split: (
                            calibration
                            if split == "calibration"
                            else predict_probabilities(
                                model,
                                tokenizer,
                                split_texts[split],
                            )
                        )
                        for split in EVALUATED_SPLITS
                    },
                }
            )

    train_model(
        tokenizer,
        texts,
        targets,
        label_keys,
        learning_rate,
        epochs,
        epochs,
        seed,
        on_epoch,
    )
    return best


def run_baseline(
    texts: list[str],
    targets: torch.Tensor,
    split_texts: dict[str, list[str]],
    split_targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(texts)
    split_features = {
        split: vectorizer.transform(split_texts[split])
        for split in EVALUATED_SPLITS
    }
    best: dict[str, Any] = {"calibration_macro_ap": -1.0}

    for c in BASELINE_C_VALUES:
        columns: dict[str, list[np.ndarray]] = {
            split: [] for split in EVALUATED_SPLITS
        }

        for label in range(targets.shape[1]):
            label_classes = targets[:, label].int().numpy()

            # A rare label can be absent from a small subset (or present in
            # all of it); its probability is then that one class.
            if len(set(label_classes)) == 1:
                for split in EVALUATED_SPLITS:
                    columns[split].append(
                        np.full(
                            split_features[split].shape[0],
                            float(label_classes[0]),
                        )
                    )

                continue

            regression = LogisticRegression(C=c, max_iter=3000)
            regression.fit(features, label_classes)

            for split in EVALUATED_SPLITS:
                columns[split].append(
                    regression.predict_proba(split_features[split])[:, 1]
                )

        probabilities = {
            split: torch.tensor(
                np.stack(columns[split], axis=1),
                dtype=torch.float32,
            )
            for split in EVALUATED_SPLITS
        }
        calibration_macro_ap = macro_ap(
            probabilities["calibration"],
            split_targets["calibration"],
        )

        if calibration_macro_ap > best["calibration_macro_ap"]:
            best = {
                "c": c,
                "calibration_macro_ap": calibration_macro_ap,
                "probabilities": probabilities,
            }

    return best


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "learningCurve.py needs a CUDA GPU; "
            "torch.cuda.is_available() is False."
        )

    if not SELECTION_PATH.exists():
        raise SystemExit(
            f"{SELECTION_PATH} not found; run src/trainClassifier.py first."
        )

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    learning_rate = selection["learning_rate"]
    label_keys = [label.machine_key for label in load_labels()]
    train_rows = load_rows(TRAIN_SPLITS)
    evaluated_rows = load_rows(set(EVALUATED_SPLITS))
    split_rows = {
        split: [row for row in evaluated_rows if row["split"] == split]
        for split in EVALUATED_SPLITS
    }
    split_texts = {
        split: [row["text"] for row in rows]
        for split, rows in split_rows.items()
    }
    split_targets = {
        split: label_targets(rows, label_keys)
        for split, rows in split_rows.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    runs = [(size, seed) for size in TRAIN_SIZES for seed in SEEDS] + [
        (len(train_rows), SEEDS[0])
    ]

    LEARNING_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEARNING_CURVE_PATH.open("w", encoding="utf-8") as curve_file:

        def write_record(record: dict[str, Any]) -> None:
            curve_file.write(json.dumps(record) + "\n")
            curve_file.flush()

        for size, seed in runs:
            order = list(range(len(train_rows)))
            random.Random(seed).shuffle(order)
            subset = [train_rows[index] for index in order[:size]]
            texts = [row["text"] for row in subset]
            targets = label_targets(subset, label_keys)

            started = time.perf_counter()
            modernbert = run_modernbert(
                tokenizer,
                texts,
                targets,
                label_keys,
                learning_rate,
                seed,
                split_texts,
                split_targets,
            )
            record = {
                "classifier": "modernbert",
                "train_size": size,
                "seed": seed,
                "learning_rate": learning_rate,
                "epochs": modernbert["epochs"],
                "evaluated_epochs": modernbert["evaluated_epochs"],
                "min_train_steps": MIN_TRAIN_STEPS,
                "best_epoch": modernbert["best_epoch"],
                "calibration_macro_ap": modernbert["calibration_macro_ap"],
                **score_run(
                    modernbert["probabilities"],
                    split_rows,
                    split_targets,
                    label_keys,
                ),
            }
            record["seconds"] = time.perf_counter() - started
            write_record(record)
            print(
                f"modernbert n={size} seed {seed}: best epoch "
                f"{record['best_epoch']}/{record['epochs']}, test macro F1 "
                f"{record['test_macro_f1']:.3f}, all-correct "
                f"{record['test_all_correct']:.3f} "
                f"({record['seconds']:.0f} s)",
                flush=True,
            )

            started = time.perf_counter()
            baseline = run_baseline(texts, targets, split_texts, split_targets)
            record = {
                "classifier": "baseline",
                "train_size": size,
                "seed": seed,
                "c_values": list(BASELINE_C_VALUES),
                "c": baseline["c"],
                "calibration_macro_ap": baseline["calibration_macro_ap"],
                **score_run(
                    baseline["probabilities"],
                    split_rows,
                    split_targets,
                    label_keys,
                ),
            }
            record["seconds"] = time.perf_counter() - started
            write_record(record)
            print(
                f"baseline n={size} seed {seed}: C={record['c']:g}, test "
                f"macro F1 {record['test_macro_f1']:.3f}, all-correct "
                f"{record['test_all_correct']:.3f} "
                f"({record['seconds']:.0f} s)",
                flush=True,
            )

    print(f"Learning curve: {LEARNING_CURVE_PATH}")


if __name__ == "__main__":
    main()
