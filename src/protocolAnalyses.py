"""
Compute the analyses the concept paper's synthetic-evaluation protocol
(Sections 4.10-4.12) asks for beyond results/content_coding/classifier_evaluation.md: core
codes, test results by story setting and by answer style, bootstrap
confidence intervals, evidence-quote fidelity, contrasting labels and error
examples.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch

from buildClassifierData import (
    ANNOTATOR_MODEL,
    DATASET_PATH,
    SECOND_RATER_MODEL,
    SPLITS,
    annotation_path,
    load_labels,
    load_rows,
)
from evaluateClassifier import (
    DEV_THRESHOLDS_PATH,
    PREDICTIONS_PATH,
    REPORT_PATH as EVALUATION_REPORT_PATH,
    binary_f1,
    kappa,
    number,
    question_number,
)
from generatePrompts import (
    Label,
    ResponseApproach,
    ResponseTone,
    load_prompt_data,
)
from trainClassifier import ROOT_DIR, label_targets, macro_ap


REPORT_PATH = ROOT_DIR / "results" / "content_coding" / "protocol_analyses.md"
# The paper's core codes (Section 4.9) as unions of codebook labels: a code is
# present when any of its labels is. AGENT_PERSPECTIVE has no codebook label.
CORE_CODES: dict[str, tuple[str, ...]] = {
    "RULE": (
        "sys_rule_justification",
        "sys_consistent_procedure",
        "sys_fairness_reciprocity",
    ),
    "CONSEQUENCE": ("sys_consequence_reasoning",),
    "CONCERN": ("ae_welfare_concern",),
    "DISTRESS": (
        "pd_harm_distress",
        "pd_decision_conflict",
        "pd_personal_consequence_anxiety",
    ),
    "AFFECTED_PERSPECTIVE": (
        "ce_mental_state_attribution",
        "ce_contextual_explanation",
    ),
}
# Questions set in the same story. The paper (Section 4.10) requires railway
# variants and the warehouse sequence to be grouped; Q12 and Q15 are the same
# split-payment job story told from opposite sides.
STORY_SETTINGS: dict[str, tuple[str, ...]] = {
    "Railway": ("Q1", "Q2", "Q3", "Q4", "Q5", "Q7", "Q8"),
    "Island rescue": ("Q6",),
    "Warehouse": ("Q9", "Q10", "Q11"),
    "Shared paid job": ("Q12", "Q15"),
    "Game exchange": ("Q13",),
    "Coworker reward": ("Q14",),
}
# Label pairs the paper asks to keep apart (Section 4.10).
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("ae_welfare_concern", "ae_shared_affect"),
    ("pda_harm_regret", "pda_self_consequence_regret"),
    ("ce_understood_harm", "ae_explicit_disregard"),
    ("ce_supportive_use", "ce_exploitative_use"),
    ("sys_rule_justification", "sys_instrumental_norm_use"),
)
ERROR_EXAMPLE_LABELS = (
    "ae_welfare_concern",
    "ae_explicit_disregard",
    "pda_harm_regret",
    "ce_exploitative_use",
)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1
# Characters translated before quote matching, and those stripped from a
# quote's ends.
PLAIN_CHARACTERS = str.maketrans(
    {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
    }
)
QUOTE_EDGE_CHARACTERS = " .\u2026\"'"

# A rater's annotation of one answer: each label key it marks present, with
# the quote that shows it.
Annotation = dict[str, str]
# Each answer's brief as an index into the sorted prompt ids, and the number
# of briefs.
Parents = tuple[torch.Tensor, int]
# A statistic per column on all answers, shaped [columns], and on each
# resample of the briefs, shaped [resamples, columns].
Resampled = tuple[torch.Tensor, torch.Tensor]
# A value, None when undefined, and its 95% interval.
Estimate = tuple[float | None, tuple[float, float]]


@dataclass(frozen=True)
class TestAnswers:
    rows: list[dict[str, Any]]
    reference: torch.Tensor
    rater: torch.Tensor
    predicted: torch.Tensor
    probabilities: torch.Tensor

    def subset(self, keep: Callable[[dict[str, Any]], bool]) -> TestAnswers:
        indices = [index for index, row in enumerate(self.rows) if keep(row)]
        return TestAnswers(
            [self.rows[index] for index in indices],
            self.reference[indices],
            self.rater[indices],
            self.predicted[indices],
            self.probabilities[indices],
        )


def story_setting(question_id: str) -> str | None:
    return next(
        (
            setting
            for setting, question_ids in STORY_SETTINGS.items()
            if question_id in question_ids
        ),
        None,
    )


def normalise(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text)
        .translate(PLAIN_CHARACTERS)
        .casefold()
        .split()
    )


@cache
def batch_annotations(path: Path) -> list[Annotation]:
    return json.loads(path.read_text(encoding="utf-8"))["annotations"]


def raw_annotation(row: dict[str, Any], annotator: str) -> Annotation:
    path = annotation_path(
        {
            "model": row["generator_model"],
            "prompt_id": row["prompt_id"],
            "batch": row["batch"],
        },
        annotator,
    )
    return batch_annotations(path)[row["response_index"] - 1]


def quote_span(quote: str, normalised_text: str) -> tuple[int, int] | None:
    normalised_quote = normalise(quote.strip(QUOTE_EDGE_CHARACTERS))

    if not normalised_quote:
        return None

    start = normalised_text.find(normalised_quote)
    return None if start < 0 else (start, start + len(normalised_quote))


def quote_counts(rows: list[dict[str, Any]], annotator: str) -> Counter:
    counts = Counter()

    for row in rows:
        normalised_text = normalise(row["text"])

        for quote in raw_annotation(row, annotator).values():
            counts["quotes"] += 1
            counts["verbatim"] += (
                bool(quote.strip(QUOTE_EDGE_CHARACTERS))
                and quote in row["text"]
            )
            counts["normalised"] += (
                quote_span(quote, normalised_text) is not None
            )

    return counts


def core_matrix(matrix: torch.Tensor, label_keys: list[str]) -> torch.Tensor:
    return torch.stack(
        [
            matrix[:, [label_keys.index(key) for key in keys]].amax(dim=1)
            for keys in CORE_CODES.values()
        ],
        dim=1,
    )


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def share(values: torch.Tensor) -> float | None:
    return values.float().mean().item() if len(values) else None


def mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def macro_f1(reference: torch.Tensor, other: torch.Tensor) -> float | None:
    scores = [
        binary_f1(reference[:, column], other[:, column])
        for column in range(reference.shape[1])
        if 0 < reference[:, column].sum() < len(reference)
    ]
    return sum(scores) / len(scores) if scores else None


def parent_groups(rows: list[dict[str, Any]]) -> Parents:
    prompt_ids = sorted({row["prompt_id"] for row in rows})
    position = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    return (
        torch.tensor([position[row["prompt_id"]] for row in rows]),
        len(prompt_ids),
    )


def parent_sums(
    parent_index: torch.Tensor,
    parent_count: int,
    values: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(
        (parent_count, *values.shape[1:]),
        dtype=torch.float64,
    ).index_add_(0, parent_index, values.double())


def parent_counts(
    parent_index: torch.Tensor,
    parent_count: int,
    reference: torch.Tensor,
    other: torch.Tensor,
) -> torch.Tensor:
    return parent_sums(
        parent_index,
        parent_count,
        torch.stack(
            (
                reference * other,
                (1 - reference) * other,
                reference * (1 - other),
                reference,
            ),
            dim=1,
        ),
    )


def f1_from_counts(
    counts: torch.Tensor,
    answers: torch.Tensor,
) -> torch.Tensor:
    true_positives, false_positives, false_negatives, positives = (
        counts.unbind(dim=-2)
    )
    f1 = 2 * true_positives / (
        2 * true_positives + false_positives + false_negatives
    )
    defined = (positives > 0) & (positives < answers.unsqueeze(-1))
    return torch.where(defined, f1, torch.nan)


def bootstrap_indices(parent_count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    return torch.randint(
        parent_count,
        (BOOTSTRAP_RESAMPLES, parent_count),
        generator=generator,
    )


def resample(per_parent: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    draws = torch.zeros(
        len(indices),
        len(per_parent),
        dtype=per_parent.dtype,
    ).scatter_add_(
        1,
        indices,
        torch.ones(indices.shape, dtype=per_parent.dtype),
    )
    return torch.tensordot(draws, per_parent, dims=1)


def interval(samples: torch.Tensor) -> tuple[float, float]:
    low, high = torch.nanquantile(
        samples,
        torch.tensor([0.025, 0.975], dtype=samples.dtype),
    ).tolist()
    return low, high


def with_interval(
    value: float | None,
    bounds: tuple[float, float],
    signed: bool = False,
) -> str:
    if value is None:
        return "–"

    low, high = bounds
    style = "+.3f" if signed else ".3f"
    return f"{value:{style}} ({low:{style}}–{high:{style}})"


def f1_estimates(
    parents: Parents,
    indices: torch.Tensor,
    reference: torch.Tensor,
    other: torch.Tensor,
) -> Resampled:
    parent_index, parent_count = parents
    counts = parent_counts(parent_index, parent_count, reference, other)
    answers = parent_sums(
        parent_index,
        parent_count,
        torch.ones(len(reference)),
    )
    return (
        f1_from_counts(counts.sum(dim=0), answers.sum()),
        f1_from_counts(resample(counts, indices), resample(answers, indices)),
    )


def paired_difference(first: Resampled, second: Resampled) -> Resampled:
    return first[0] - second[0], first[1] - second[1]


def macro_estimate(statistic: Resampled) -> Estimate:
    point, samples = statistic
    return torch.nanmean(point).item(), interval(torch.nanmean(samples, dim=1))


def column_estimate(statistic: Resampled, column: int) -> Estimate:
    point, samples = statistic
    value = None if point[column].isnan() else point[column].item()
    return value, interval(samples[:, column])


def share_estimate(
    parents: Parents,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> Estimate:
    parent_index, parent_count = parents
    sums = parent_sums(
        parent_index,
        parent_count,
        torch.stack((values, torch.ones_like(values)), dim=1),
    )
    resampled = resample(sums, indices)
    return (
        (sums[:, 0].sum() / sums[:, 1].sum()).item(),
        interval(resampled[:, 0] / resampled[:, 1]),
    )


def core_codes_section(
    labels: list[Label],
    reference: torch.Tensor,
    rater: torch.Tensor,
    predicted: torch.Tensor,
    classifier_f1: Resampled,
    rater_f1: Resampled,
) -> list[str]:
    names = {label.machine_key: label.label_name for label in labels}
    true_positives = (reference * predicted).sum(dim=0).tolist()
    marked = predicted.sum(dim=0).tolist()
    positives = reference.sum(dim=0).tolist()
    precisions = [
        ratio(hits, count) for hits, count in zip(true_positives, marked)
    ]
    recalls = [
        ratio(hits, count) for hits, count in zip(true_positives, positives)
    ]
    classifier_kappas = [
        kappa(reference[:, column], predicted[:, column])
        for column in range(len(CORE_CODES))
    ]
    rater_kappas = [
        kappa(reference[:, column], rater[:, column])
        for column in range(len(CORE_CODES))
    ]
    lines = [
        "## Core codes",
        "",
        f"The paper's core codes (Section 4.9) as unions of codebook labels: "
        f"a code is present in an answer when any of its labels is. "
        f"AGENT_PERSPECTIVE has no codebook label and is not scored. "
        f"Precision, recall, F1 and κ (Cohen's kappa) compare the "
        f"classifier's predictions and the second rater's labels with "
        f"`{ANNOTATOR_MODEL}`'s.",
        "",
        f"| Code | Codebook labels | Answers | `{ANNOTATOR_MODEL}` positives "
        f"| Precision | Recall | Classifier F1 (95% CI) | Classifier κ "
        f"| Rater F1 (95% CI) | Rater κ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for column, (code, keys) in enumerate(CORE_CODES.items()):
        lines.append(
            f"| {code} | {', '.join(names[key] for key in keys)} "
            f"| {len(reference):,} | {int(reference[:, column].sum()):,} "
            f"| {number(precisions[column])} | {number(recalls[column])} "
            f"| {with_interval(*column_estimate(classifier_f1, column))} "
            f"| {number(classifier_kappas[column])} "
            f"| {with_interval(*column_estimate(rater_f1, column))} "
            f"| {number(rater_kappas[column])} |"
        )

    lines.append(
        f"| **Macro average** | – | – | – "
        f"| {number(mean_defined(precisions))} "
        f"| {number(mean_defined(recalls))} "
        f"| {with_interval(*macro_estimate(classifier_f1))} "
        f"| {number(mean_defined(classifier_kappas))} "
        f"| {with_interval(*macro_estimate(rater_f1))} "
        f"| {number(mean_defined(rater_kappas))} |"
    )
    return lines


def setting_row(
    name: str,
    subset: TestAnswers,
    parents: Parents,
    indices: torch.Tensor,
) -> str:
    questions = sorted(
        {row["question_id"] for row in subset.rows},
        key=question_number,
    )
    classifier = macro_estimate(
        f1_estimates(parents, indices, subset.reference, subset.predicted)
    )
    rater = macro_estimate(
        f1_estimates(parents, indices, subset.reference, subset.rater)
    )
    all_correct = share((subset.reference == subset.predicted).all(dim=1))
    return (
        f"| {name} | {', '.join(questions)} | {len(subset.rows):,} "
        f"| {with_interval(*classifier)} "
        f"| {number(macro_ap(subset.probabilities, subset.reference))} "
        f"| {number(all_correct)} | {with_interval(*rater)} |"
    )


def story_setting_section(
    rows: list[dict[str, Any]],
    test: TestAnswers,
    trained_settings: set[str],
    parents: Parents,
    indices: torch.Tensor,
) -> list[str]:
    question_splits = {
        row["question_id"]: row["split"]
        for row in rows
        if row["split"] != "cross_generator"
    }
    lines = [
        "## Test questions by story setting",
        "",
        "Story settings group the questions set in the same story. Every test "
        "question is held out whole. Under the paper's Table 5 content "
        "families, all four test questions share a family with a training "
        "question, so this is not the family holdout the protocol specifies. "
        "It separates the test questions whose story setting appears in "
        "training from those whose setting does not. Macro AP uses the saved "
        "probabilities, which are rounded to four decimals; the ties this "
        "creates lower it slightly against the main report.",
        "",
        "| Story setting | Train | Calibration | Test |",
        "|---|---|---|---|",
    ]

    for setting, question_ids in STORY_SETTINGS.items():
        cells = [
            ", ".join(
                sorted(
                    (
                        question_id
                        for question_id in question_ids
                        if question_splits.get(question_id) == split
                    ),
                    key=question_number,
                )
            )
            or "–"
            for split in ("train", "calibration", "test")
        ]
        lines.append(f"| {setting} | {' | '.join(cells)} |")

    lines += [
        "",
        "| Test questions | Questions | Answers "
        "| Classifier macro F1 (95% CI) | Macro AP | All correct "
        "| Second-rater macro F1 (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for name, seen in (
        ("Setting seen in training", True),
        ("New setting", False),
    ):
        subset = test.subset(
            lambda row: (story_setting(row["question_id"]) in trained_settings)
            == seen
        )

        if subset.rows:
            subset_parents = parent_groups(subset.rows)
            lines.append(
                setting_row(
                    name,
                    subset,
                    subset_parents,
                    bootstrap_indices(subset_parents[1]),
                )
            )

    lines.append(setting_row("All test questions", test, parents, indices))
    return lines


def answer_style_section(
    test: TestAnswers,
    approaches: dict[str, ResponseApproach],
    tones: dict[str, ResponseTone],
) -> list[str]:
    lines = [
        "## Test by answer style",
        "",
        "Each brief set one approach and one tone for its answers; the test "
        "answers are grouped by their brief's.",
        "",
        "| Answer style | Answers | Classifier macro F1 | All correct "
        "| Second-rater macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    styles = [
        (f"Approach: {approach.label}", "approach_id", approach.id)
        for approach in approaches.values()
    ] + [
        (f"Tone: {tone.label}", "tone_id", tone.id)
        for tone in tones.values()
    ]

    for name, field, style_id in styles:
        subset = test.subset(lambda row: row[field] == style_id)
        all_correct = share((subset.reference == subset.predicted).all(dim=1))
        lines.append(
            f"| {name} | {len(subset.rows):,} "
            f"| {number(macro_f1(subset.reference, subset.predicted))} "
            f"| {number(all_correct)} "
            f"| {number(macro_f1(subset.reference, subset.rater))} |"
        )

    return lines


def confidence_section(
    labels: list[Label],
    test: TestAnswers,
    parents: Parents,
    indices: torch.Tensor,
    classifier_f1: Resampled,
    rater_f1: Resampled,
    core_classifier_f1: Resampled,
    core_rater_f1: Resampled,
) -> list[str]:
    difference = paired_difference(classifier_f1, rater_f1)
    classifier_correct = test.predicted == test.reference
    headline = [
        ("Classifier macro F1", macro_estimate(classifier_f1), False),
        ("Second-rater macro F1", macro_estimate(rater_f1), False),
        (
            "Classifier minus second-rater macro F1 (paired)",
            macro_estimate(difference),
            True,
        ),
        (
            "Classifier all correct",
            share_estimate(
                parents,
                indices,
                classifier_correct.all(dim=1).float(),
            ),
            False,
        ),
        (
            "Second rater all agree",
            share_estimate(
                parents,
                indices,
                (test.rater == test.reference).all(dim=1).float(),
            ),
            False,
        ),
        (
            "Classifier pair accuracy",
            share_estimate(
                parents,
                indices,
                classifier_correct.float().mean(dim=1),
            ),
            False,
        ),
        (
            "Core-code classifier macro F1",
            macro_estimate(core_classifier_f1),
            False,
        ),
        (
            "Core-code second-rater macro F1",
            macro_estimate(core_rater_f1),
            False,
        ),
    ]
    lines = [
        "## Confidence intervals",
        "",
        f"Intervals resample the test briefs as described above. The "
        f"classifier and the second rater are compared with "
        f"`{ANNOTATOR_MODEL}` on the same resamples, so their difference is "
        f"paired.",
        "",
        "| Measure | Value (95% CI) |",
        "|---|---:|",
    ]
    lines += [
        f"| {name} | {with_interval(*estimate, signed=signed)} |"
        for name, estimate, signed in headline
    ]
    lines += [
        "",
        "| Label | Positives | Classifier F1 (95% CI) | Rater F1 (95% CI) "
        "| Classifier − rater (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    above = 0
    below: list[str] = []

    for column, label in enumerate(labels):
        gap, (low, high) = column_estimate(difference, column)
        lines.append(
            f"| {label.label_name} | {int(test.reference[:, column].sum()):,} "
            f"| {with_interval(*column_estimate(classifier_f1, column))} "
            f"| {with_interval(*column_estimate(rater_f1, column))} "
            f"| {with_interval(gap, (low, high), signed=True)} |"
        )

        if gap is not None and low > 0:
            above += 1
        elif gap is not None and high < 0:
            below.append(label.label_name)

    names = f" ({', '.join(below)})" if below else ""
    lines += [
        "",
        f"The interval of the difference lies wholly above zero for {above} "
        f"labels, wholly below zero for {len(below)}{names}, and includes "
        f"zero for {len(labels) - above - len(below)}.",
    ]
    return lines


def evidence_section(
    labels: list[Label],
    test_rows: list[dict[str, Any]],
    fidelity: list[tuple[str, str, int, Counter]],
) -> list[str]:
    lines = [
        "## Evidence quotes",
        "",
        "Each rater quotes the shortest phrase, at most 15 words, that shows "
        "each characteristic it marks present. Verbatim means the quote "
        "occurs exactly in the answer. After normalisation means it occurs "
        "once both are normalised: NFKC, casefolding, collapsed whitespace, "
        "plain quote marks and dashes, and the quote's surrounding spaces, "
        "full stops, ellipses and quote marks stripped. An empty quote "
        "counts as not found. Mental-state attributions added only as a "
        "prerequisite of another label carry no quote and are not counted.",
        "",
        "| Rater | Split | Answers | Quotes | Verbatim | After normalisation "
        "| Not found |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for annotator, split, answers, counts in fidelity:
        lines.append(
            f"| `{annotator}` | {split} | {answers:,} | {counts['quotes']:,} "
            f"| {number(ratio(counts['verbatim'], counts['quotes']))} "
            f"| {number(ratio(counts['normalised'], counts['quotes']))} "
            f"| {counts['quotes'] - counts['normalised']:,} |"
        )

    group_of = {label.machine_key: label.group for label in labels}
    pairs = Counter()
    located = Counter()
    overlapping = Counter()

    for row in test_rows:
        normalised_text = normalise(row["text"])
        reference_quotes = raw_annotation(row, ANNOTATOR_MODEL)
        rater_quotes = raw_annotation(row, SECOND_RATER_MODEL)

        for key in reference_quotes.keys() & rater_quotes.keys():
            group = group_of[key]
            pairs[group] += 1
            first = quote_span(reference_quotes[key], normalised_text)
            second = quote_span(rater_quotes[key], normalised_text)

            if first is not None and second is not None:
                located[group] += 1
                overlapping[group] += max(first[0], second[0]) < min(
                    first[1], second[1]
                )

    lines += [
        "",
        "Where both raters mark the same characteristic in a test answer, "
        "each quote is located at its first occurrence in the normalised "
        "answer; the two spans overlap when they share at least one "
        "character. Spans overlap is the share of the pairs with both quotes "
        "located.",
        "",
        "| Domain | Pairs both raters mark | Both quotes located "
        "| Spans overlap |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {group} | {pairs[group]:,} | {located[group]:,} "
        f"| {number(ratio(overlapping[group], located[group]))} |"
        for group in dict.fromkeys(label.group for label in labels)
    ]
    lines.append(
        f"| **All** | {pairs.total():,} | {located.total():,} "
        f"| {number(ratio(overlapping.total(), located.total()))} |"
    )
    return lines


def contrast_section(
    labels: list[Label],
    label_keys: list[str],
    test: TestAnswers,
) -> list[str]:
    names = {label.machine_key: label.label_name for label in labels}
    lines = [
        "## Contrasting labels",
        "",
        f"The label pairs the paper asks to keep apart (Section 4.10). For "
        f"the answers where `{ANNOTATOR_MODEL}` marks the first label but not "
        f"the second: how often the classifier finds the first, how often it "
        f"wrongly predicts the second, and how often `{SECOND_RATER_MODEL}` "
        f"marks the second; then the same with the labels swapped. "
        f"Unprovoked versus defensive noncooperation has no codebook labels.",
        "",
        "| Contrast | Only first: answers | First found | Second predicted "
        "| Rater marks second | Only second: answers | Second found "
        "| First predicted | Rater marks first |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for first, second in CONTRASTS:
        cells: list[str] = []

        for present, absent in (
            (label_keys.index(first), label_keys.index(second)),
            (label_keys.index(second), label_keys.index(first)),
        ):
            only = (test.reference[:, present] == 1) & (
                test.reference[:, absent] == 0
            )
            cells += [
                f"{int(only.sum()):,}",
                number(share(test.predicted[only, present])),
                number(share(test.predicted[only, absent])),
                number(share(test.rater[only, absent])),
            ]

        lines.append(
            f"| *{names[first]}* vs *{names[second]}* | {' | '.join(cells)} |"
        )

    return lines


def rater_status(
    row: dict[str, Any],
    key: str,
    annotator: str,
    field: str,
) -> str:
    quotes = raw_annotation(row, annotator)

    if key in quotes:
        return f"present, quoting “{quotes[key]}”"

    if key in row[field]:
        return "present (implied by another label)"

    return "absent"


def error_examples_section(
    labels: list[Label],
    label_keys: list[str],
    test: TestAnswers,
    thresholds: dict[str, float],
) -> list[str]:
    names = {label.machine_key: label.label_name for label in labels}
    lines = [
        "## Error examples",
        "",
        "All answers are synthetic, written by the named model to a brief. "
        "For each label: the test answer the classifier misses with the "
        "lowest probability, and the answer it wrongly marks with the "
        "highest probability. Ties go to the lower id.",
    ]

    for key in ERROR_EXAMPLE_LABELS:
        column = label_keys.index(key)
        probabilities = test.probabilities[:, column].tolist()
        reference = test.reference[:, column].tolist()
        predicted = test.predicted[:, column].tolist()

        for heading, error, sign in (
            ("missed", (1, 0), 1),
            ("wrongly predicted", (0, 1), -1),
        ):
            candidates = [
                index
                for index in range(len(test.rows))
                if (reference[index], predicted[index]) == error
            ]
            lines += ["", f"### {names[key]}: {heading}", ""]

            if not candidates:
                lines.append("None in the test split.")
                continue

            index = min(
                candidates,
                key=lambda candidate: (
                    sign * probabilities[candidate],
                    test.rows[candidate]["id"],
                ),
            )
            row = test.rows[index]
            statuses = [
                f"- `{annotator}`: {rater_status(row, key, annotator, field)}"
                for annotator, field in (
                    (ANNOTATOR_MODEL, "annotated_label_keys"),
                    (SECOND_RATER_MODEL, "second_rater_label_keys"),
                )
            ]
            lines += [
                f"- Answer: `{row['id']}` ({row['question_id']}, written by "
                f"`{row['generator_model']}`)",
                f"- Classifier probability {probabilities[index]:.3f}, "
                f"threshold {thresholds[key]:.2f}",
                *statuses,
                "",
                f"> {row['text']}",
            ]

    return lines


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(
            f"{PREDICTIONS_PATH} not found; run src/evaluateClassifier.py "
            f"first."
        )

    labels = load_labels()
    label_keys = [label.machine_key for label in labels]
    _, _, approaches, tones = load_prompt_data()
    rows = load_rows(set(SPLITS))
    rows_by_id = {row["id"]: row for row in rows}

    with PREDICTIONS_PATH.open(encoding="utf-8") as predictions_file:
        predictions = [json.loads(line) for line in predictions_file]

    test_predictions = [
        prediction
        for prediction in predictions
        if prediction["split"] == "test"
    ]
    test_ids = {row["id"] for row in rows if row["split"] == "test"}
    stale = any(
        prediction["id"] not in rows_by_id for prediction in predictions
    ) or {prediction["id"] for prediction in test_predictions} != test_ids

    if stale:
        raise SystemExit(
            f"{PREDICTIONS_PATH} does not match {DATASET_PATH}; rerun "
            f"src/evaluateClassifier.py."
        )

    for question_id in sorted(
        {row["question_id"] for row in rows},
        key=question_number,
    ):
        if story_setting(question_id) is None:
            raise SystemExit(f"{question_id} has no entry in STORY_SETTINGS.")

    thresholds = json.loads(DEV_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    test_rows = [
        rows_by_id[prediction["id"]] for prediction in test_predictions
    ]
    test = TestAnswers(
        test_rows,
        label_targets(test_rows, label_keys),
        label_targets(test_rows, label_keys, "second_rater_label_keys"),
        label_targets(test_predictions, label_keys, "predicted_label_keys"),
        torch.tensor(
            [
                [prediction["probabilities"][key] for key in label_keys]
                for prediction in test_predictions
            ]
        ),
    )
    parents = parent_groups(test.rows)
    indices = bootstrap_indices(parents[1])
    classifier_f1 = f1_estimates(
        parents,
        indices,
        test.reference,
        test.predicted,
    )
    rater_f1 = f1_estimates(parents, indices, test.reference, test.rater)
    core_reference = core_matrix(test.reference, label_keys)
    core_rater = core_matrix(test.rater, label_keys)
    core_predicted = core_matrix(test.predicted, label_keys)
    core_classifier_f1 = f1_estimates(
        parents,
        indices,
        core_reference,
        core_predicted,
    )
    core_rater_f1 = f1_estimates(parents, indices, core_reference, core_rater)
    trained_settings = {
        story_setting(row["question_id"])
        for row in rows
        if row["split"] == "train"
    }
    new_setting = test.subset(
        lambda row: story_setting(row["question_id"]) not in trained_settings
    )
    fidelity = [
        (
            annotator,
            split,
            len(split_rows),
            quote_counts(split_rows, annotator),
        )
        for annotator, split, split_rows in (
            (ANNOTATOR_MODEL, "all", rows),
            (ANNOTATOR_MODEL, "test", test.rows),
            (SECOND_RATER_MODEL, "test", test.rows),
        )
    ]
    lines = [
        "# Protocol analyses",
        "",
        f"These are the analyses the concept paper's synthetic-evaluation "
        f"protocol (Sections 4.10–4.12) asks for beyond "
        f"`{EVALUATION_REPORT_PATH.relative_to(ROOT_DIR)}`. They use the dev "
        f"model's test predictions at its calibration-tuned thresholds "
        f"(`{PREDICTIONS_PATH.relative_to(ROOT_DIR)}`), with "
        f"`{ANNOTATOR_MODEL}`'s labels as the reference and "
        f"`{SECOND_RATER_MODEL}` as the second rater. Intervals are 95% "
        f"percentile intervals from {BOOTSTRAP_RESAMPLES:,} resamples of the "
        f"{parents[1]:,} test briefs, the paper's parent groups, with all "
        f"answers to a brief resampled together.",
        "",
        *core_codes_section(
            labels,
            core_reference,
            core_rater,
            core_predicted,
            core_classifier_f1,
            core_rater_f1,
        ),
        "",
        *story_setting_section(rows, test, trained_settings, parents, indices),
        "",
        *answer_style_section(test, approaches, tones),
        "",
        *confidence_section(
            labels,
            test,
            parents,
            indices,
            classifier_f1,
            rater_f1,
            core_classifier_f1,
            core_rater_f1,
        ),
        "",
        *evidence_section(labels, test.rows, fidelity),
        "",
        *contrast_section(labels, label_keys, test),
        "",
        *error_examples_section(labels, label_keys, test, thresholds),
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reference_quotes, rater_quotes = fidelity[1][3], fidelity[2][3]
    print(
        f"Test macro F1 {with_interval(*macro_estimate(classifier_f1))}; "
        f"second rater {with_interval(*macro_estimate(rater_f1))}; "
        f"core codes {with_interval(*macro_estimate(core_classifier_f1))}; "
        f"new-setting macro F1 "
        f"{number(macro_f1(new_setting.reference, new_setting.predicted))}; "
        f"verbatim quotes "
        f"{reference_quotes['verbatim'] / reference_quotes['quotes']:.3f} / "
        f"{rater_quotes['verbatim'] / rater_quotes['quotes']:.3f}. "
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
