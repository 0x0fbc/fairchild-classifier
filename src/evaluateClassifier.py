"""
Tune decision thresholds for the trained classifiers and evaluate them.
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from buildClassifierData import (
    ANNOTATOR_MODEL,
    SECOND_RATER_MODEL,
    SPLITS,
    TEST_QUESTIONS,
    TRAINING_GENERATORS,
    load_labels,
    load_rows,
    write_jsonl,
)
from generatePrompts import Label
from generateResponses import write_json
from trainClassifier import (
    DEV_MODEL_DIR,
    EPOCHS,
    LEARNING_CURVE_PATH,
    MODEL_DIR,
    ROOT_DIR,
    SELECTION_PATH,
    label_targets,
    predict_probabilities,
)


# Thresholds for the shipped model in MODEL_DIR.
THRESHOLDS_PATH = MODEL_DIR.parent / "thresholds.json"
# Thresholds and calibration, test and cross_generator predictions of the dev
# model in DEV_MODEL_DIR.
DEV_THRESHOLDS_PATH = DEV_MODEL_DIR.parent / "thresholds.json"
PREDICTIONS_PATH = ROOT_DIR / "results" / "content_coding" / "predictions.jsonl"
REPORT_PATH = ROOT_DIR / "results" / "content_coding" / "classifier_evaluation.md"
EVALUATED_SPLITS = ("calibration", "test", "cross_generator")
MIN_OBSERVED_FOR_THRESHOLD = 5
# Candidate thresholds in hundredths: 0.01 to 0.99.
THRESHOLD_HUNDREDTHS = range(1, 100)
METRIC_FIELDS = ("accuracy", "precision", "recall", "f1", "ap", "auc")
METRICS_NOTE = (
    "Per label, accuracy is the share of answers predicted correctly. Recall "
    "needs positives, and precision, F1, AP and AUC need both positives and "
    "negatives; otherwise they show –, and macro averages leave them out. All "
    "correct is the share of answers with every label right."
)
# Second rater table columns: F1 and Cohen's kappa against the annotation.
AGREEMENT_FIELDS = (
    "rater_f1",
    "classifier_f1",
    "rater_kappa",
    "classifier_kappa",
)
# learningCurve.py records grouped by (classifier, train_size).
LearningCurve = dict[tuple[str, int], list[dict[str, Any]]]


@dataclass(frozen=True)
class Predictions:

    rows: list[dict[str, Any]]
    probabilities: torch.Tensor
    targets: torch.Tensor

    def subset(self, keep: Callable[[dict[str, Any]], bool]) -> Predictions:
        indices = [index for index, row in enumerate(self.rows) if keep(row)]
        return Predictions(
            [self.rows[index] for index in indices],
            self.probabilities[indices],
            self.targets[indices],
        )


def question_number(question_id: str) -> int:
    return int(question_id.removeprefix("Q"))


def load_classifier(
    model_dir: Path,
    label_keys: list[str],
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if not model_dir.exists():
        raise SystemExit(
            f"{model_dir} not found; run src/trainClassifier.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(
        "cuda"
    )
    model_keys = [
        model.config.id2label[index] for index in range(model.config.num_labels)
    ]

    if model_keys != label_keys:
        raise SystemExit(
            f"{model_dir} labels differ from materials/codebook.csv; "
            "run src/trainClassifier.py again."
        )

    return model, tokenizer


def predict_splits(
    model_dir: Path,
    label_keys: list[str],
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, torch.Tensor]:
    model, tokenizer = load_classifier(model_dir, label_keys)
    probabilities = {
        split: predict_probabilities(
            model,
            tokenizer,
            [row["text"] for row in rows],
        )
        for split, rows in split_rows.items()
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return probabilities


def choose_threshold(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    positives = int(targets.sum())
    negatives = len(targets) - positives

    if min(positives, negatives) < MIN_OBSERVED_FOR_THRESHOLD:
        return 0.5

    is_positive = targets == 1

    def f1(hundredths: int) -> float:
        predicted = probabilities >= hundredths / 100
        true_positives = int((predicted & is_positive).sum())
        false_positives = int((predicted & ~is_positive).sum())
        false_negatives = positives - true_positives
        return 2 * true_positives / (
            2 * true_positives + false_positives + false_negatives
        )

    best = max(
        THRESHOLD_HUNDREDTHS,
        key=lambda hundredths: (f1(hundredths), -abs(hundredths - 50), -hundredths),
    )
    return best / 100


def tune_thresholds(predictions: Predictions) -> list[float]:
    return [
        choose_threshold(
            predictions.probabilities[:, label],
            predictions.targets[:, label],
        )
        for label in range(predictions.targets.shape[1])
    ]


def label_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    true_labels = targets.int().numpy()
    scores = probabilities.numpy()
    positives = int(true_labels.sum())
    negatives = len(true_labels) - positives
    metrics: dict[str, Any] = {
        "positives": positives,
        "negatives": negatives,
        "threshold": threshold,
    } | dict.fromkeys(METRIC_FIELDS)

    if positives + negatives == 0:
        return metrics

    predicted = (scores >= threshold).astype(int)
    metrics["accuracy"] = float((predicted == true_labels).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted,
        average="binary",
        zero_division=0,
    )

    if positives:
        metrics["recall"] = float(recall)

    if positives and negatives:
        metrics |= {
            "precision": float(precision),
            "f1": float(f1),
            "ap": float(average_precision_score(true_labels, scores)),
            "auc": float(roc_auc_score(true_labels, scores)),
        }

    return metrics


def all_label_metrics(
    predictions: Predictions,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    return [
        label_metrics(
            predictions.probabilities[:, label],
            predictions.targets[:, label],
            threshold,
        )
        for label, threshold in enumerate(thresholds)
    ]


def macro(metrics: list[dict[str, Any]], field: str) -> float | None:
    values = [entry[field] for entry in metrics if entry[field] is not None]
    return sum(values) / len(values) if values else None


def row_scores(
    predictions: Predictions,
    thresholds: torch.Tensor,
) -> tuple[float, float]:
    correct = (predictions.probabilities >= thresholds) == (
        predictions.targets == 1
    )
    all_correct = correct.all(dim=1).float().mean().item()
    pair_accuracy = correct.float().mean().item()
    return all_correct, pair_accuracy


def binary_f1(reference: torch.Tensor, other: torch.Tensor) -> float | None:
    denominator = int(reference.sum() + other.sum())
    true_positives = int((reference * other).sum())
    return 2 * true_positives / denominator if denominator else None


def kappa(reference: torch.Tensor, other: torch.Tensor) -> float | None:
    observed = (reference == other).float().mean().item()
    reference_rate = reference.mean().item()
    other_rate = other.mean().item()
    chance = reference_rate * other_rate + (1 - reference_rate) * (
        1 - other_rate
    )

    if chance == 1:
        return None

    return (observed - chance) / (1 - chance)


def agreement_metrics(
    targets: torch.Tensor,
    rater: torch.Tensor,
    predicted: torch.Tensor,
) -> list[dict[str, Any]]:
    return [
        {
            "annotator_positives": int(targets[:, label].sum()),
            "rater_positives": int(rater[:, label].sum()),
            "rater_f1": binary_f1(targets[:, label], rater[:, label]),
            "classifier_f1": binary_f1(targets[:, label], predicted[:, label]),
            "rater_kappa": kappa(targets[:, label], rater[:, label]),
            "classifier_kappa": kappa(targets[:, label], predicted[:, label]),
            "classifier_f1_against_rater": binary_f1(
                rater[:, label],
                predicted[:, label],
            ),
        }
        for label in range(targets.shape[1])
    ]


def rater_level_count(agreement: list[dict[str, Any]]) -> int:
    return sum(
        entry["classifier_f1"] >= entry["rater_f1"]
        for entry in agreement
        if entry["classifier_f1"] is not None
        and entry["rater_f1"] is not None
    )


def number(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def difference(value: float | None, reference: float | None) -> str:
    if value is None or reference is None:
        return "–"

    return f"{value - reference:+.3f}"


def summary(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"

    return (
        f"{sum(values) / len(values):.3f} "
        f"({min(values):.3f}–{max(values):.3f})"
    )


def label_table(
    labels: list[Label],
    metrics: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "| Label | Positives | Negatives | Threshold | Accuracy | Precision "
        "| Recall | F1 | AP | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for label, entry in zip(labels, metrics):
        scores = " | ".join(number(entry[field]) for field in METRIC_FIELDS)
        lines.append(
            f"| {label.label_name} | {entry['positives']} "
            f"| {entry['negatives']} | {number(entry['threshold'])} "
            f"| {scores} |"
        )

    macros = " | ".join(number(macro(metrics, field)) for field in METRIC_FIELDS)
    lines.append(f"| **Macro average** | – | – | – | {macros} |")
    return lines


def rows_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Split | "
        + " | ".join(f"`{model}`" for model in TRAINING_GENERATORS)
        + " | Other generators | Total |",
        "|---|" + "---:|" * (len(TRAINING_GENERATORS) + 2),
    ]

    for split in SPLITS:
        counts = Counter(
            row["generator_model"] for row in rows if row["split"] == split
        )
        others = sum(counts.values()) - sum(
            counts[model] for model in TRAINING_GENERATORS
        )
        cells = [str(counts[model]) for model in TRAINING_GENERATORS]
        lines.append(
            f"| {split} | {' | '.join(cells)} | {others} "
            f"| {sum(counts.values())} |"
        )

    return lines


def design_counts(
    rows: list[dict[str, Any]],
    label_keys: list[str],
) -> dict[str, Counter]:
    counts = {key: Counter() for key in label_keys}

    for row in rows:
        positives = set(row["positive_label_keys"])
        negatives = set(row["negative_label_keys"])
        annotated = set(row["annotated_label_keys"])

        for key in label_keys:
            if key in positives:
                kind = "positive"
            elif key in negatives:
                kind = "negative"
            else:
                kind = "open"

            counts[key][kind] += 1
            counts[key][kind, key in annotated] += 1

    return counts


def share(counts: Counter, kind: str, marked: bool) -> str:
    total = counts[kind]
    return number(counts[kind, marked] / total if total else None)


def agreement_cells(counts: Counter) -> str:
    return (
        f"{counts['positive']} | {share(counts, 'positive', True)} "
        f"| {counts['negative']} | {share(counts, 'negative', False)} "
        f"| {counts['open']} | {share(counts, 'open', True)}"
    )


def load_learning_curve(train_size: int) -> list[dict[str, Any]]:
    if not LEARNING_CURVE_PATH.exists():
        raise SystemExit(
            f"{LEARNING_CURVE_PATH} not found; run src/learningCurve.py first."
        )

    with LEARNING_CURVE_PATH.open(encoding="utf-8") as curve_file:
        records = [json.loads(line) for line in curve_file]

    full_size = {
        record["classifier"]
        for record in records
        if record["train_size"] == train_size
    }

    if not {"modernbert", "baseline"} <= full_size:
        raise SystemExit(
            f"{LEARNING_CURVE_PATH} has no full-size run; run "
            f"src/learningCurve.py again."
        )

    return records


def curve_sizes(curve: LearningCurve) -> list[int]:
    return sorted({size for _, size in curve})


def mean_macro_f1(curve: LearningCurve, size: int) -> float:
    records = curve["modernbert", size]
    return sum(record["test_macro_f1"] for record in records) / len(records)


def summary_section(
    test_questions: list[str],
    test_answers: int,
    test_metrics: list[dict[str, Any]],
    test_all_correct: float,
    cross_all_correct: float,
    agreement: list[dict[str, Any]],
    rater_all_agree: float,
    curve: LearningCurve,
    train_size: int,
) -> list[str]:
    full_f1 = mean_macro_f1(curve, train_size)
    baseline = curve["baseline", train_size][0]
    sizes = curve_sizes(curve)
    near_full = next(
        size for size in sizes if mean_macro_f1(curve, size) >= 0.95 * full_f1
    )
    beating = [
        size
        for size in sizes
        if mean_macro_f1(curve, size) > baseline["test_macro_f1"]
    ]

    if beating:
        baseline_clause = (
            f"with {beating[0]:,} it already beats the baseline trained on "
            f"all {train_size:,} ({mean_macro_f1(curve, beating[0]):.3f} "
            f"against {baseline['test_macro_f1']:.3f})."
        )
    else:
        baseline_clause = (
            f"it does not beat the baseline trained on all {train_size:,}."
        )

    return [
        "## Summary",
        "",
        f"- Held-out dilemmas ({', '.join(test_questions)}): macro F1 "
        f"{number(macro(test_metrics, 'f1'))}, macro AP "
        f"{number(macro(test_metrics, 'ap'))}, every label right in "
        f"{number(test_all_correct)} of {test_answers:,} answers.",
        f"- Word-count baseline trained on the same {train_size:,} answers: "
        f"macro F1 {number(baseline['test_macro_f1'])}, every label right in "
        f"{number(baseline['test_all_correct'])}. On unseen generators' "
        f"answers every label is right in {number(cross_all_correct)} for "
        f"the classifier and "
        f"{number(baseline['cross_generator_all_correct'])} for the baseline.",
        f"- Second rater `{SECOND_RATER_MODEL}` against `{ANNOTATOR_MODEL}` "
        f"on the same answers: macro F1 "
        f"{number(macro(agreement, 'rater_f1'))}, every label agreeing in "
        f"{number(rater_all_agree)}. The classifier's F1 is at least the "
        f"second rater's on {rater_level_count(agreement)} of "
        f"{len(agreement)} labels.",
        f"- Learning curve: with {near_full:,} labelled answers the "
        f"classifier reaches {mean_macro_f1(curve, near_full):.3f} macro F1, "
        f"at least 95% of the {full_f1:.3f} it reaches with all "
        f"{train_size:,}; {baseline_clause}",
    ]


def baseline_section(
    labels: list[Label],
    test_metrics: list[dict[str, Any]],
    test_all_correct: float,
    cross_all_correct: float,
    baseline: dict[str, Any],
    train_size: int,
) -> list[str]:
    lines = [
        "## Word-count baseline",
        "",
        f"A TF-IDF bag of words and word pairs with one logistic regression "
        f"per label, trained on the same {train_size:,} answers, with its "
        f"regularisation and thresholds chosen on the calibration split like "
        f"the classifier's. It shows how much of each label word choice "
        f"alone reveals.",
        "",
        "| Label | Positives | Baseline F1 | Classifier F1 | Difference |",
        "|---|---:|---:|---:|---:|",
    ]

    for label, entry in zip(labels, test_metrics):
        baseline_f1 = baseline["label_f1"][label.machine_key]
        lines.append(
            f"| {label.label_name} | {entry['positives']} "
            f"| {number(baseline_f1)} | {number(entry['f1'])} "
            f"| {difference(entry['f1'], baseline_f1)} |"
        )

    classifier_f1 = macro(test_metrics, "f1")
    lines += [
        f"| **Macro average** | – | {number(baseline['test_macro_f1'])} "
        f"| {number(classifier_f1)} "
        f"| {difference(classifier_f1, baseline['test_macro_f1'])} |",
        "",
        f"Every label right: baseline {number(baseline['test_all_correct'])}, "
        f"classifier {number(test_all_correct)}. Unseen generators' answers: "
        f"baseline {number(baseline['cross_generator_all_correct'])}, "
        f"classifier {number(cross_all_correct)}.",
    ]
    return lines


def second_rater_section(
    labels: list[Label],
    agreement: list[dict[str, Any]],
    test_answers: int,
    rater_all_agree: float,
    rater_pair_agreement: float,
    test_all_correct: float,
    test_pair_accuracy: float,
) -> list[str]:
    lines = [
        "## Second rater",
        "",
        f"`{SECOND_RATER_MODEL}` annotated the {test_answers:,} test answers "
        f"with the same prompt as `{ANNOTATOR_MODEL}`, standing in for a "
        f"second human rater. With human labels, a classifier is as reliable "
        f"as a rater once it agrees with that rater about as well as a "
        f"second rater does. Rater columns compare `{SECOND_RATER_MODEL}` "
        f"with `{ANNOTATOR_MODEL}`; classifier columns compare the dev "
        f"model's predictions at its thresholds with `{ANNOTATOR_MODEL}`, "
        f"whose labels it was trained on. F1 is symmetric in the two sides; "
        f"κ is Cohen's kappa, which discounts chance agreement.",
        "",
        f"| Label | `{ANNOTATOR_MODEL}` positives | `{SECOND_RATER_MODEL}` "
        f"positives | Rater F1 | Classifier F1 | Rater κ | Classifier κ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for label, entry in zip(labels, agreement):
        scores = " | ".join(number(entry[field]) for field in AGREEMENT_FIELDS)
        lines.append(
            f"| {label.label_name} | {entry['annotator_positives']} "
            f"| {entry['rater_positives']} | {scores} |"
        )

    shortfalls = sorted(
        (
            (label, entry)
            for label, entry in zip(labels, agreement)
            if entry["classifier_f1"] is not None
            and entry["rater_f1"] is not None
            and entry["classifier_f1"] < entry["rater_f1"]
        ),
        key=lambda item: item[1]["rater_f1"] - item[1]["classifier_f1"],
        reverse=True,
    )
    comparison = (
        f"The classifier's F1 is at least the second rater's on "
        f"{rater_level_count(agreement)} of {len(labels)} labels"
    )

    if shortfalls:
        comparison += "; it falls furthest short on " + ", ".join(
            f"*{label.label_name}* ({entry['classifier_f1']:.3f} against "
            f"{entry['rater_f1']:.3f})"
            for label, entry in shortfalls[:3]
        )

    macros = " | ".join(
        number(macro(agreement, field)) for field in AGREEMENT_FIELDS
    )
    lines += [
        f"| **Macro average** | – | – | {macros} |",
        "",
        f"Every label agrees with `{ANNOTATOR_MODEL}` in "
        f"{number(rater_all_agree)} of answers for the second rater and "
        f"{number(test_all_correct)} for the classifier; pair agreement "
        f"{number(rater_pair_agreement)} and {number(test_pair_accuracy)}.",
        "",
        f"Against `{SECOND_RATER_MODEL}`, whose labels it never saw, the "
        f"classifier reaches macro F1 "
        f"{number(macro(agreement, 'classifier_f1_against_rater'))}.",
        "",
        f"{comparison}.",
    ]
    return lines


def learning_curve_section(
    labels: list[Label],
    curve: LearningCurve,
    train_size: int,
    calibration_answers: int,
) -> list[str]:
    sizes = curve_sizes(curve)
    full = curve["modernbert", train_size][0]
    c_values = curve["baseline", train_size][0]["c_values"]

    def cell(classifier: str, size: int, field: str) -> str:
        return summary([record[field] for record in curve[classifier, size]])

    lines = [
        "## Learning curve",
        "",
        f"Each size is a random subset of the {train_size:,} training "
        f"answers, drawn {len(curve['modernbert', sizes[0]])} times with "
        f"different seeds below the full set; cells show the mean with the "
        f"range in brackets. The classifier trains at learning rate "
        f"{full['learning_rate']:g} for at least {EPOCHS} epochs and "
        f"{full['min_train_steps']:,} optimizer steps (Epochs column), "
        f"keeping the best of {len(full['evaluated_epochs'])} evenly spaced "
        f"checkpoints by calibration macro AP; the baseline picks C from "
        f"{', '.join(f'{c:g}' for c in c_values)} the same way. Every run "
        f"picks its checkpoint and thresholds on the full calibration split "
        f"({calibration_answers:,} answers), which a real labelling budget "
        f"would also have to cover.",
        "",
        "| Training answers | Epochs | Classifier macro F1 "
        "| Classifier all correct | Baseline macro F1 "
        "| Baseline all correct |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for size in sizes:
        lines.append(
            f"| {size:,} | {curve['modernbert', size][0]['epochs']} "
            f"| {cell('modernbert', size, 'test_macro_f1')} "
            f"| {cell('modernbert', size, 'test_all_correct')} "
            f"| {cell('baseline', size, 'test_macro_f1')} "
            f"| {cell('baseline', size, 'test_all_correct')} |"
        )

    lines += [
        "",
        "Classifier F1 per label (mean over seeds):",
        "",
        "| Label | " + " | ".join(f"{size:,}" for size in sizes) + " |",
        "|---|" + "---:|" * len(sizes),
    ]

    for label in labels:
        cells: list[str] = []

        for size in sizes:
            values = [
                record["label_f1"][label.machine_key]
                for record in curve["modernbert", size]
                if record["label_f1"][label.machine_key] is not None
            ]
            mean = sum(values) / len(values) if values else None
            cells.append(number(mean))

        lines.append(f"| {label.label_name} | {' | '.join(cells)} |")

    macros = " | ".join(f"{mean_macro_f1(curve, size):.3f}" for size in sizes)
    lines.append(f"| **Macro average** | {macros} |")
    return lines


def limits_section(
    labels: list[Label],
    train_rows: list[dict[str, Any]],
    generators: str,
) -> list[str]:
    shares = (
        label_targets(train_rows, [label.machine_key for label in labels])
        .mean(dim=0)
        .tolist()
    )
    rarest = min(range(len(labels)), key=shares.__getitem__)
    commonest = max(range(len(labels)), key=shares.__getitem__)
    return [
        "## Limits of the synthetic data",
        "",
        f"- The answers were written to briefs by {generators} that required "
        f"or forbade particular cues, so they are fluent, on topic and state "
        f"their cues plainly. Human answers will be less tidy, and scores on "
        f"them are expected to be lower.",
        f"- The briefs make every cue common: in the training answers the "
        f"rater marks each label present in between {shares[rarest]:.1%} "
        f"(*{labels[rarest].label_name}*) and {shares[commonest]:.1%} "
        f"(*{labels[commonest].label_name}*) of answers. In human answers "
        f"some cues are likely to be much rarer, and a rare cue needs more "
        f"labelled answers than the learning curve above shows.",
        "- Both raters are language models, so the agreement figures measure "
        "agreement between models, not with human judgement; two models can "
        "share blind spots that human raters would not.",
        "- The learning curve and the second-rater comparison are the parts "
        "that carry over to human data: repeated on a human-labelled sample, "
        "they show how many answers to label and whether the classifier "
        "agrees with a rater as well as a second rater does.",
    ]


def main() -> None:
    labels = load_labels()
    label_keys = [label.machine_key for label in labels]
    rows = load_rows(set(SPLITS))
    train_rows = [row for row in rows if row["split"] == "train"]
    curve: LearningCurve = {}

    for record in load_learning_curve(len(train_rows)):
        curve.setdefault(
            (record["classifier"], record["train_size"]),
            [],
        ).append(record)

    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in EVALUATED_SPLITS
    }
    dev_probabilities = predict_splits(DEV_MODEL_DIR, label_keys, split_rows)
    dev = {
        split: Predictions(
            split_rows[split],
            dev_probabilities[split],
            label_targets(split_rows[split], label_keys),
        )
        for split in EVALUATED_SPLITS
    }
    calibration = dev["calibration"]
    test = dev["test"]
    cross_generator = dev["cross_generator"]
    thresholds = tune_thresholds(calibration)
    write_json(DEV_THRESHOLDS_PATH, dict(zip(label_keys, thresholds)))
    threshold_tensor = torch.tensor(thresholds)

    shipped_probabilities = predict_splits(
        MODEL_DIR,
        label_keys,
        {"calibration": split_rows["calibration"]},
    )["calibration"]
    shipped_calibration = Predictions(
        calibration.rows,
        shipped_probabilities,
        calibration.targets,
    )
    shipped_thresholds = tune_thresholds(shipped_calibration)
    write_json(THRESHOLDS_PATH, dict(zip(label_keys, shipped_thresholds)))

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    test_questions = sorted(TEST_QUESTIONS, key=question_number)
    generators = " and ".join(f"`{model}`" for model in TRAINING_GENERATORS)
    test_metrics = all_label_metrics(test, thresholds)
    test_all_correct, test_pair_accuracy = row_scores(test, threshold_tensor)
    cross_metrics = all_label_metrics(cross_generator, thresholds)
    cross_all_correct, cross_pair_accuracy = row_scores(
        cross_generator,
        threshold_tensor,
    )
    rater = label_targets(test.rows, label_keys, "second_rater_label_keys")
    agreement = agreement_metrics(
        test.targets,
        rater,
        (test.probabilities >= threshold_tensor).float(),
    )
    rater_agrees = rater == test.targets
    rater_all_agree = rater_agrees.all(dim=1).float().mean().item()
    rater_pair_agreement = rater_agrees.float().mean().item()
    baseline = curve["baseline", len(train_rows)][0]
    lines = [
        "# Classifier evaluation",
        "",
        f"This report tests whether a simple pipeline, fine-tuning a "
        f"pretrained encoder on rater-labelled answers, yields a classifier "
        f"that finds the {len(labels)} reasoning cues defined in "
        f"`materials/codebook.csv`. The answers are synthetic, written to briefs "
        f"by {generators}, and every label of every answer comes from "
        f"`{ANNOTATOR_MODEL}`, which did not see the briefs; they stand in "
        f"for human answers and human raters. The briefs' required and "
        f"forbidden labels are used only to check the data (see *Annotation "
        f"agreement with the prompt design*).",
        "",
        f"Dev model: `{selection['base_model']}` fine-tuned on the train "
        f"dilemmas at learning rate {selection['learning_rate']:g}, saved "
        f"after epoch {selection['epoch']}, chosen by calibration macro AP "
        f"{number(selection['calibration_macro_ap'])} (calibration loss "
        f"{number(selection['calibration_loss'])}). Max length "
        f"{selection['max_length']} tokens, batch size "
        f"{selection['batch_size']}, seed {selection['seed']}.",
        "",
        f"The shipped model, `{MODEL_DIR.relative_to(ROOT_DIR)}/` with "
        f"`{THRESHOLDS_PATH.relative_to(ROOT_DIR)}`, retrains with these "
        f"settings on the train and test dilemmas (calibration macro AP "
        f"{number(selection['final_calibration_macro_ap'])}), so the test "
        f"scores below describe the dev model.",
        "",
        *rows_table(rows),
        "",
        METRICS_NOTE,
        "",
        *summary_section(
            test_questions,
            len(test.rows),
            test_metrics,
            test_all_correct,
            cross_all_correct,
            agreement,
            rater_all_agree,
            curve,
            len(train_rows),
        ),
        "",
        f"## Test: held-out dilemmas {', '.join(test_questions)}",
        "",
        *label_table(labels, test_metrics),
        "",
        f"All labels correct in {number(test_all_correct)} of rows; pair "
        f"accuracy {number(test_pair_accuracy)} over "
        f"{test.targets.numel()} pairs.",
        "",
    ]

    for generator_model in TRAINING_GENERATORS:
        generator = test.subset(
            lambda row: row["generator_model"] == generator_model
        )

        if generator.rows:
            all_correct, pair_accuracy = row_scores(generator, threshold_tensor)
            lines.append(
                f"- `{generator_model}`: {len(generator.rows)} rows, pair "
                f"accuracy {number(pair_accuracy)}, all labels correct in "
                f"{number(all_correct)} of rows."
            )

    lines += [
        "",
        *baseline_section(
            labels,
            test_metrics,
            test_all_correct,
            cross_all_correct,
            baseline,
            len(train_rows),
        ),
        "",
        *second_rater_section(
            labels,
            agreement,
            len(test.rows),
            rater_all_agree,
            rater_pair_agreement,
            test_all_correct,
            test_pair_accuracy,
        ),
        "",
        *learning_curve_section(
            labels,
            curve,
            len(train_rows),
            len(calibration.rows),
        ),
        "",
        "## Test by dilemma",
        "",
        "| Dilemma | Rows | Macro F1 | Macro AP | All correct |",
        "|---|---:|---:|---:|---:|",
    ]

    for question_id in test_questions:
        question = test.subset(lambda row: row["question_id"] == question_id)
        question_metrics = all_label_metrics(question, thresholds)
        all_correct, _ = row_scores(question, threshold_tensor)
        lines.append(
            f"| {question_id} | {len(question.rows)} "
            f"| {number(macro(question_metrics, 'f1'))} "
            f"| {number(macro(question_metrics, 'ap'))} "
            f"| {number(all_correct)} |"
        )

    cross_prompts = sorted(
        {row["prompt_id"] for row in cross_generator.rows},
        key=lambda prompt_id: (
            question_number(prompt_id.split("_")[0]),
            prompt_id,
        ),
    )
    lines += [
        "",
        f"## Unseen generators: held-out prompts {', '.join(cross_prompts)}",
        "",
        "| Generator | Rows | Pair accuracy | All correct |",
        "|---|---:|---:|---:|",
    ]

    for generator_model in sorted(
        {row["generator_model"] for row in cross_generator.rows}
    ):
        generator = cross_generator.subset(
            lambda row: row["generator_model"] == generator_model
        )
        all_correct, pair_accuracy = row_scores(generator, threshold_tensor)
        lines.append(
            f"| {generator_model} | {len(generator.rows)} "
            f"| {number(pair_accuracy)} | {number(all_correct)} |"
        )

    calibration_metrics = all_label_metrics(calibration, thresholds)
    calibration_all_correct, _ = row_scores(calibration, threshold_tensor)
    shipped_metrics = all_label_metrics(shipped_calibration, shipped_thresholds)
    shipped_all_correct, _ = row_scores(
        shipped_calibration,
        torch.tensor(shipped_thresholds),
    )
    lines += [
        f"| **All unseen generators** | {len(cross_generator.rows)} "
        f"| {number(cross_pair_accuracy)} | {number(cross_all_correct)} |",
        "",
        "Pooled over all unseen generators:",
        "",
        *label_table(labels, cross_metrics),
        "",
        "## Calibration",
        "",
        f"Dev model: macro F1 {number(macro(calibration_metrics, 'f1'))}, "
        f"macro AP {number(macro(calibration_metrics, 'ap'))}, all labels "
        f"correct in {number(calibration_all_correct)} of rows. Shipped "
        f"model: macro F1 {number(macro(shipped_metrics, 'f1'))}, macro AP "
        f"{number(macro(shipped_metrics, 'ap'))}, all labels correct in "
        f"{number(shipped_all_correct)} of rows. These scores are optimistic, "
        f"because each model's thresholds were tuned on this set.",
        "",
        "## Annotation agreement with the prompt design",
        "",
        "The prompt design is not used for training or scoring; it checks the "
        "data. For each label: the designed positives (labels a brief "
        "required) the annotation marks present, the designed negatives "
        "(labels a brief forbade) it marks absent, and the open pairs it "
        "marks present.",
        "",
        "| Label | Designed positives | Marked present | Designed negatives "
        "| Marked absent | Open pairs | Open marked present |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    label_counts = design_counts(rows, label_keys)

    for label in labels:
        lines.append(
            f"| {label.label_name} "
            f"| {agreement_cells(label_counts[label.machine_key])} |"
        )

    all_counts = sum(label_counts.values(), Counter())
    lines += [
        f"| **All labels** | {agreement_cells(all_counts)} |",
        "",
        "All labels, by generator:",
        "",
    ]

    for generator_model in sorted({row["generator_model"] for row in rows}):
        generator_rows = [
            row for row in rows if row["generator_model"] == generator_model
        ]
        counts = sum(
            design_counts(generator_rows, label_keys).values(),
            Counter(),
        )
        lines.append(
            f"- `{generator_model}`: {share(counts, 'positive', True)} of "
            f"{counts['positive']} designed positives marked present, "
            f"{share(counts, 'negative', False)} of {counts['negative']} "
            f"designed negatives marked absent, {share(counts, 'open', True)} "
            f"of {counts['open']} open pairs marked present."
        )

    lines += ["", *limits_section(labels, train_rows, generators)]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prediction_rows: list[dict[str, Any]] = []

    for split_predictions in (calibration, test, cross_generator):
        predicted = split_predictions.probabilities >= threshold_tensor

        for row, probabilities, row_predicted in zip(
            split_predictions.rows,
            split_predictions.probabilities.tolist(),
            predicted.tolist(),
        ):
            prediction_rows.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "generator_model": row["generator_model"],
                    "probabilities": {
                        key: round(probability, 4)
                        for key, probability in zip(label_keys, probabilities)
                    },
                    "predicted_label_keys": [
                        key
                        for key, is_predicted in zip(label_keys, row_predicted)
                        if is_predicted
                    ],
                    "positive_label_keys": row["positive_label_keys"],
                    "negative_label_keys": row["negative_label_keys"],
                    "annotated_label_keys": row["annotated_label_keys"],
                    "second_rater_label_keys": row["second_rater_label_keys"],
                }
            )

    write_jsonl(PREDICTIONS_PATH, prediction_rows)
    print(
        f"Test macro F1 {macro(test_metrics, 'f1'):.3f}, "
        f"macro AP {macro(test_metrics, 'ap'):.3f}, "
        f"all-correct {test_all_correct:.3f}; "
        f"unseen generators all-correct {cross_all_correct:.3f}; "
        f"baseline macro F1 {baseline['test_macro_f1']:.3f}; "
        f"second rater macro F1 {macro(agreement, 'rater_f1'):.3f}. "
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
