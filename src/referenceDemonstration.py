"""Demonstrate Appendix C's reference patterns using the frozen dev classifier.

Run templateContrastSet.py to completion first. Only the contrast answers need
GPU inference; held-out answers reuse the dev predictions without retraining.
The report preserves neither-pattern, processing review, and empty denominators.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

import torch
from sklearn.metrics import cohen_kappa_score

from annotateResponses import completion_cost
from buildClassifierData import (
    ANNOTATOR_MODEL, DATASET_PATH, SECOND_RATER_MODEL, annotation_path,
    load_labels, load_rows, question_split,
)
from evaluateClassifier import (
    DEV_THRESHOLDS_PATH, PREDICTIONS_PATH, binary_f1, number,
    predict_splits, question_number,
)
from protocolAnalyses import (
    CORE_CODES, bootstrap_indices, column_estimate, f1_estimates,
    parent_groups, ratio, share, with_interval,
)
from referencePatterns import (
    A_ONLY, B_ONLY, INSUFFICIENT, INVALID, MISSING, MIXED, NEITHER, OUTCOMES,
    REVIEW, DOMAINS, ReportDomain, TEMPLATES, aggregate,
    domain_outcome, gated_outcome, template_met, with_prerequisites,
)
from templateContrastSet import (
    TEMPLATE_ANNOTATIONS_ROOT, TEMPLATE_PROMPTS_PATH, contrast_batches,
    contrast_rows, eligible_questions,
)
from trainClassifier import DEV_MODEL_DIR, ROOT_DIR


REPORT_PATH = ROOT_DIR / "results" / "reference_patterns" / "reference_demonstration.md"
ADMINISTRATIONS = 1000
ADMINISTRATION_SEED = 1
TEMPLATE_DOMAINS = tuple(d for d in DOMAINS if d.a_templates or d.b_templates)
SOURCE_NAMES = {"test": "Test questions", "calibration": "Calibration questions", "contrast": "Contrast set"}


@dataclass(frozen=True)
class Answer:
    row: dict[str, Any]
    reference: frozenset[str]
    rater: frozenset[str] | None
    predicted: frozenset[str]
    probabilities: dict[str, float]


@dataclass(frozen=True)
class Record:
    source: str
    row: dict[str, Any]
    domain: ReportDomain
    reference_outcome: str
    rater_outcome: str | None
    classifier_outcome: str
    gated: str


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows), "",
    ]


def percent(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def questions_text(questions) -> str:
    return ", ".join(sorted(questions, key=question_number)) or "–"


def boolean_share(values: list[bool]) -> float | None:
    return share(torch.tensor(values, dtype=torch.bool))


def mean_defined(values) -> float | None:
    defined = [v for v in values if v is not None]
    return mean(defined) if defined else None


def engine_checks() -> list[str]:
    fields = ("n", "d", "a_allocation", "b_allocation", "indeterminate", "invalid", "coverage", "review", "missing")
    checks = [
        ("C6 example", [A_ONLY] * 3 + [B_ONLY] + [MIXED] * 2 + [NEITHER, INVALID], (8, 6, 66.7, 33.3, 12.5, 12.5, 75.0, 0.0, 0.0)),
        ("All Indeterminate", [NEITHER] * 8, (8, 0, None, None, 100.0, 0.0, 0.0, 0.0, 0.0)),
        ("All Mixed", [MIXED] * 8, (8, 8, 50.0, 50.0, 0.0, 0.0, 100.0, 0.0, 0.0)),
        ("All Invalid", [INVALID] * 8, (8, 0, None, None, 0.0, 100.0, 0.0, 0.0, 0.0)),
        ("Empty", [], (0, 0, None, None, None, None, None, None, None)),
    ]
    def describe(values):
        n, d, a, b, ind, invalid, coverage, review, missing = values
        return f"N {n}, D {d}, A {percent(a)}, B {percent(b)}, Indeterminate {percent(ind)}, Invalid {percent(invalid)}, coverage {percent(coverage)}, review {percent(review)}, missing {percent(missing)}"
    rows = []
    for name, statuses, expected in checks:
        result = aggregate(statuses)
        computed = tuple(None if (v := getattr(result, field)) is None else round(v, 1) for field in fields)
        if computed != expected:
            raise SystemExit(f"Engine check failed: {name}: expected {expected}, computed {computed}")
        assert sum(result.counts.values()) == result.n
        counts = ", ".join(f"{key}: {value}" for key, value in result.counts.items())
        rows.append([name, counts, describe(expected), describe(computed)])
    return ["## Engine checks", "", *table(["Check", "Records", "Expected", "Computed"], rows), "In every check the displayed counts sum to N.", ""]


def make_answer(row, keys, probabilities) -> Answer:
    return Answer(
        row, frozenset(row["annotated_label_keys"]),
        None if row["second_rater_label_keys"] is None else frozenset(row["second_rater_label_keys"]),
        with_prerequisites(keys), probabilities,
    )


def held_out_answers() -> list[Answer]:
    rows = load_rows({"calibration", "test"})
    with PREDICTIONS_PATH.open(encoding="utf-8") as stream:
        predictions = {p["id"]: p for p in map(json.loads, stream) if p["split"] in {"calibration", "test"}}
    if set(predictions) != {row["id"] for row in rows}:
        raise SystemExit(f"{PREDICTIONS_PATH} does not match {DATASET_PATH}; rerun src/evaluateClassifier.py.")
    return [make_answer(row, predictions[row["id"]]["predicted_label_keys"], predictions[row["id"]]["probabilities"]) for row in rows]


def outcome_records(answers: list[Answer], thresholds) -> list[Record]:
    return [
        Record(
            answer.row["split"], answer.row, domain,
            domain_outcome(domain, answer.reference),
            None if answer.rater is None else domain_outcome(domain, answer.rater),
            domain_outcome(domain, answer.predicted),
            gated_outcome(domain, answer.predicted, answer.probabilities, thresholds),
        )
        for answer in answers for domain in TEMPLATE_DOMAINS
        if answer.row["question_id"] in domain.questions
    ]


def agreement(records: list[Record], field: str) -> float | None:
    pairs = [(r.reference_outcome, getattr(r, field)) for r in records if getattr(r, field) is not None]
    return ratio(sum(a == b for a, b in pairs), len(pairs))


def outcome_kappa(records: list[Record], field: str) -> float | None:
    pairs = [(r.reference_outcome, getattr(r, field)) for r in records if getattr(r, field) is not None]
    if not pairs:
        return None
    reference, other = zip(*pairs)
    # A single shared outcome has chance agreement 1 and undefined kappa.
    if len(set(reference) | set(other)) == 1:
        return None
    value = float(cohen_kappa_score(reference, other, labels=list(OUTCOMES)))
    return None if math.isnan(value) else value


def review_share(records: list[Record]) -> float | None:
    return ratio(sum(r.gated == REVIEW for r in records), len(records))


def resolved_agreement(records: list[Record]) -> float | None:
    return agreement([r for r in records if r.gated != REVIEW], "gated")


def support_f1(records: list[Record], field: str) -> list[str]:
    records = [r for r in records if getattr(r, field) is not None]
    if not records:
        return ["–", "–"]
    parents = parent_groups([r.row for r in records])
    indices = bootstrap_indices(parents[1])
    def columns(field):
        return torch.tensor([
            [getattr(r, field) in (A_ONLY, MIXED), getattr(r, field) in (B_ONLY, MIXED)]
            for r in records
        ], dtype=torch.float32)
    estimate = f1_estimates(parents, indices, columns("reference_outcome"), columns(field))
    return [with_interval(*column_estimate(estimate, c)) for c in range(2)]


def templates_section(labels) -> list[str]:
    names = {label.machine_key: label.label_name for label in labels}
    rows = []
    for template in TEMPLATES.values():
        conjunction = [names[key] for key in sorted(template.all_of)]
        conjunction += ["(" + " or ".join(names[key] for key in sorted(group)) + ")" for group in template.any_of]
        domains = ", ".join(d.name for d in TEMPLATE_DOMAINS if template.id in d.a_templates + d.b_templates)
        rows.append([template.id, template.name, " + ".join(conjunction), domains, template.note or "–"])
    return ["## Templates", "", *table(["Template", "Name", "Codebook conjunction", "Registered domains", "Note"], rows),
            "All conjunctions are checked at the answer level: raters and classifier mark each characteristic present or absent for the whole answer, so the same-person and same-time links of Table C2 are not verified.", ""]


def registry_section(answers: list[Answer]) -> list[str]:
    questions = {source: {a.row["question_id"] for a in answers if a.row["split"] == source} for source in SOURCE_NAMES}
    rows = [[
        d.name + (" (descriptive only)" if d.id == "distress" else ""), questions_text(d.questions),
        ", ".join(d.a_templates) or "–", ", ".join(d.b_templates) or "–",
        *(questions_text(d.questions & questions[source]) for source in SOURCE_NAMES),
    ] for d in DOMAINS]
    return ["## Registry", "", *table(["Domain", "Eligible questions", "A templates", "B templates", "Test questions", "Calibration questions", "Contrast-set questions"], rows)]


def contrast_section(answers: list[Answer], batches) -> list[str]:
    briefs = [json.loads(line) for line in TEMPLATE_PROMPTS_PATH.read_text(encoding="utf-8").splitlines()]
    concern = next(d for d in DOMAINS if d.id == "concern")
    rows = []
    for template_id in (*TEMPLATES, "A2+B1"):
        for case in ("mixed",) if template_id == "A2+B1" else ("positive", "near_miss"):
            selected_briefs = [b for b in briefs if (b["template_id"], b["case"]) == (template_id, case)]
            selected_batches = [b for b in batches if (b["template_id"], b["case"]) == (template_id, case)]
            selected = [a for a in answers if a.row["split"] == "contrast" and (a.row["template_id"], a.row["case"]) == (template_id, case)]
            matches = [
                domain_outcome(concern, a.reference) == MIXED if case == "mixed"
                else template_met(TEMPLATES[template_id], a.reference) == (case == "positive")
                for a in selected
            ]
            rows.append([template_id, case, len(selected_briefs), len(selected_batches), sum(b["status"] == "incompatible" for b in selected_batches), len(selected), number(boolean_share(matches))])
    generators = sorted({b["model"] for b in batches})
    generation_costs = {m: sum(completion_cost(b) for b in batches if b["model"] == m) for m in generators}
    annotation_costs = {
        m: sum(completion_cost(json.loads(annotation_path(b, m, root=TEMPLATE_ANNOTATIONS_ROOT).read_text(encoding="utf-8"))) for b in batches if b["status"] == "ok")
        for m in (ANNOTATOR_MODEL, SECOND_RATER_MODEL)
    }
    cost_text = "; ".join(f"{m} ${v:.2f}" for m, v in generation_costs.items())
    annotation_text = "; ".join(f"{m} ${v:.2f}" for m, v in annotation_costs.items())
    total_answers = sum(b["status"] == "ok" for b in batches) * 10
    declined = sum(b["status"] == "incompatible" for b in batches)
    return ["## Contrast set", "",
            "Briefs target full conjunctions and near misses on each template's registered questions. A1 substitutes contextual mental-state explanation for mentalizing difficulty; A2 substitutes ordinary rule justification for a consistent procedure; A3 includes responsibility and repair variants, with procedure omitted in their near misses. B1's near miss uses consequence reasoning without concern or disregard; B2 contrasts exploitative use with contextual consequence reasoning; B3 contrasts disregard with concern while retaining instrumental norm use; B4 contrasts explicit absence of harm regret with harm regret while retaining self-consequence regret. B2 and B4 each have two positive and two near-miss briefs per eligible question. Mixed briefs require a consistent procedure and concern for one identified person alongside disregard for another, following Table C5's Q05 example. These are separately generated answers, not paired edits; some near misses remove more than one element.", "",
            "The generators were `claude-opus-5-5` at medium effort and `claude-sonnet-5` at default effort, one batch of 10 answers per brief per generator. Both raters used the standard characteristic-annotation prompt, without design labels or direct template-level annotation. Declined batches were excluded and not re-prompted.", "",
            f"Total: {len(briefs)} briefs, {len(batches)} batches, {declined} declined batches and {total_answers} answers.", "",
            *table(["Template", "Case", "Briefs", "Batches", "Declined", "Answers", "Rater-1 agreement with design"], rows),
            f"Generation cost: {cost_text}. Annotation cost: {annotation_text}. Total saved-completion cost: ${sum(generation_costs.values()) + sum(annotation_costs.values()):.2f}.", ""]


def agreement_section(records: list[Record]) -> list[str]:
    lines = ["## Outcome agreement", "",
             "Agreement and κ use ungated outcomes against rater 1. Support F1 intervals resample whole briefs (2,000 draws; 95% percentile intervals). F1 is undefined when the reference has no positive or no negative. Review uses a ±0.10 shift of every threshold, clipped to 0.01–0.99; resolved agreement excludes review. All-domains rows pool eligible answer–domain records, not independent answers. All shares are fractions.", ""]
    for source, name in SOURCE_NAMES.items():
        lines += [f"### {name}", ""]
        if source == "calibration":
            lines += ["Thresholds were tuned on these questions, so these scores are optimistic.", ""]
        selected = [r for r in records if r.source == source]
        rows, rater_rows = [], []
        for domain in (*TEMPLATE_DOMAINS, None):
            subset = [r for r in selected if domain is None or r.domain == domain]
            counts = Counter(r.reference_outcome for r in subset)
            name = "**All domains**" if domain is None else domain.name
            f1 = ["–", "–"] if domain is None else support_f1(subset, "classifier_outcome")
            rows.append([
                name, len(subset), " / ".join(str(counts[s]) for s in OUTCOMES),
                number(agreement(subset, "classifier_outcome")), "–" if domain is None else number(outcome_kappa(subset, "classifier_outcome")),
                *f1, number(review_share(subset)), number(resolved_agreement(subset)),
                number(agreement(subset, "rater_outcome")), "–" if domain is None else number(outcome_kappa(subset, "rater_outcome")),
            ])
            if domain is not None:
                rater_rows.append([name, *support_f1(subset, "rater_outcome")])
        lines += table(["Domain", "Records", "Rater-1 A-only / B-only / Mixed / Indeterminate", "Classifier agreement", "Classifier κ", "Support A F1 (95% CI)", "Support B F1 (95% CI)", "Review share", "Agreement when resolved", "Rater-2 agreement", "Rater-2 κ"], rows)
        lines += table(["Domain", "Rater-2 support A F1 (95% CI)", "Rater-2 support B F1 (95% CI)"], rater_rows)
    return lines


def template_section(answers: list[Answer]) -> list[str]:
    lines = ["## Templates against rater 1", ""]
    for source in ("test", "contrast"):
        lines += [f"### {SOURCE_NAMES[source]}", ""]
        rows = []
        for template_id, template in TEMPLATES.items():
            eligible = eligible_questions(template_id)
            selected = [a for a in answers if a.row["split"] == source and a.row["question_id"] in eligible]
            reference = torch.tensor([template_met(template, a.reference) for a in selected], dtype=torch.float32).reshape(-1, 1)
            predicted = torch.tensor([template_met(template, a.predicted) for a in selected], dtype=torch.float32).reshape(-1, 1)
            f1s = ["–", "–"]
            if selected:
                parents = parent_groups([a.row for a in selected])
                indices = bootstrap_indices(parents[1])
                rater = torch.tensor([template_met(template, a.rater) for a in selected], dtype=torch.float32).reshape(-1, 1)
                estimates = [column_estimate(f1_estimates(parents, indices, reference, other), 0) for other in (predicted, rater)]
                f1s = [with_interval(*estimate) for estimate in estimates]
                if 0 < reference.sum() < len(reference):
                    assert math.isclose(estimates[0][0], binary_f1(reference, predicted))
            rows.append([template_id, len(selected), int(reference.sum()), int(predicted.sum()), *f1s])
        lines += table(["Template", "Answers assessed", "Rater-1 matches", "Classifier matches", "Classifier F1 (95% CI)", "Rater-2 F1 (95% CI)"], rows)
    return lines


def confusion_section(records: list[Record]) -> list[str]:
    lines = ["## Contrast-set confusion", ""]
    for domain in TEMPLATE_DOMAINS:
        counts = Counter((r.reference_outcome, r.gated) for r in records if r.source == "contrast" and r.domain == domain)
        rows = [["Indeterminate" if reference == NEITHER else reference, *(counts[reference, outcome] for outcome in (*OUTCOMES, REVIEW))] for reference in OUTCOMES]
        lines += [f"### {domain.name}", "", *table(["Rater 1 \\ Classifier", "A-only", "B-only", "Mixed", "Indeterminate", "Processing review"], rows)]
    return lines


def familiarity_section(records: list[Record]) -> list[str]:
    rows = []
    for split, name in (("train", "Training questions (new answers)"), ("calibration", "Calibration questions"), ("test", "Test questions")):
        subset = [r for r in records if r.source == "contrast" and question_split(r.row["question_id"]) == split]
        rows.append([name, len(subset), number(agreement(subset, "classifier_outcome")), number(review_share(subset)), number(resolved_agreement(subset)), number(agreement(subset, "rater_outcome"))])
    return ["## Contrast set by question familiarity", "", *table(["Questions", "Records", "Classifier agreement", "Review share", "Agreement when resolved", "Rater-2 agreement"], rows)]


def indeterminate_section(records: list[Record]) -> list[str]:
    lines = ["## Indeterminate rates", ""]
    for name, sources in (("Held-out corpus", {"test", "calibration"}), ("Contrast set", {"contrast"})):
        lines += [f"### {name}", ""]
        rows = []
        for domain in TEMPLATE_DOMAINS:
            for question in sorted(domain.questions, key=question_number):
                subset = [r for r in records if r.source in sources and r.domain == domain and r.row["question_id"] == question]
                if subset:
                    rows.append([domain.name, question, len(subset), number(boolean_share([r.reference_outcome == NEITHER for r in subset])), number(boolean_share([r.classifier_outcome == NEITHER for r in subset])), number(review_share(subset))])
        lines += table(["Domain", "Question", "Records", "Rater-1 Indeterminate", "Classifier Indeterminate", "Review share"], rows)
    test = [r for r in records if r.source == "test"]
    neither = sum(r.reference_outcome == NEITHER for r in test)
    lines += [f"On test questions, rater-1 Indeterminate records were {neither} / {len(test)} = {number(ratio(neither, len(test)))} ({percent(100 * neither / len(test))}%).", "",
              "There were no probes (prompt count 0 throughout) and no Invalid records. Insufficient reasoning was not assessed, so there were no insufficient-reasoning records.", ""]
    return lines


def distress_section(answers: list[Answer]) -> list[str]:
    domain = next(d for d in DOMAINS if d.id == "distress")
    rows = []
    for name, sources in (("Held-out corpus", {"test", "calibration"}), ("Contrast set", {"contrast"})):
        selected = [a for a in answers if a.row["split"] in sources and a.row["question_id"] in domain.questions]
        values = []
        for keys in (frozenset(CORE_CODES["DISTRESS"]), frozenset({"pd_explicit_low_distress"})):
            values.append(" / ".join(number(boolean_share([bool(getattr(a, field) & keys) for a in selected if getattr(a, field) is not None])) for field in ("reference", "predicted", "rater")))
        rows.append([name, len(selected), *values])
    return ["## Personal distress", "", "Descriptive only; no A/B allocation or directional Indeterminate outcome is defined. Rater 2 in the held-out corpus covers test answers only, not calibration answers. Cells are prevalence shares.", "",
            *table(["Source", "Records", "DISTRESS expressed: rater 1 / classifier / rater 2", "Explicitly low distress: rater 1 / classifier / rater 2"], rows)]


def administration_section(answers: list[Answer], records: list[Record]) -> list[str]:
    lookup = {(r.row["id"], r.domain.id): r for r in records}
    summary, example = [], []
    for source, sources in (("Held-out corpus", {"test", "calibration"}), ("Contrast set", {"contrast"})):
        by_question = defaultdict(list)
        for answer in answers:
            if answer.row["split"] in sources:
                by_question[answer.row["question_id"]].append(answer)
        by_question = {q: sorted(by_question[q], key=lambda a: a.row["id"]) for q in sorted(by_question, key=question_number)}
        rng = random.Random(f"{ADMINISTRATION_SEED}:{source}")
        aggregates = {d.id: [] for d in TEMPLATE_DOMAINS}
        for index in range(ADMINISTRATIONS):
            chosen = [rng.choice(options) for options in by_question.values()]
            for domain in TEMPLATE_DOMAINS:
                selected = [lookup[a.row["id"], domain.id] for a in chosen if a.row["question_id"] in domain.questions]
                pair = (aggregate([r.reference_outcome for r in selected]), aggregate([r.gated for r in selected]))
                aggregates[domain.id].append(pair)
                if source == "Contrast set" and index == 0:
                    for name, result in zip(("Rater 1", "Classifier"), pair, strict=True):
                        c = result.counts
                        example.append([domain.name, name, result.n, c[A_ONLY], c[B_ONLY], c[MIXED], f"{c[NEITHER]} / {c[INSUFFICIENT]}", c[INVALID], c[REVIEW], c[MISSING], result.d, percent(result.a_allocation), percent(result.b_allocation), percent(result.indeterminate), percent(result.invalid), percent(result.coverage)])
        for domain in TEMPLATE_DOMAINS:
            pairs = aggregates[domain.id]
            n = pairs[0][0].n
            if not n:
                summary.append([source, domain.name, "not assessed", *(["–"] * 6)])
                continue
            both = [(a.a_allocation, b.a_allocation) for a, b in pairs if a.d and b.d]
            summary.append([
                source, domain.name, n,
                " / ".join(number(ratio(sum(pair[i].d == 0 for pair in pairs), ADMINISTRATIONS)) for i in range(2)),
                " / ".join(percent(mean_defined(pair[i].coverage for pair in pairs)) for i in range(2)),
                " / ".join(percent(mean_defined(pair[i].a_allocation for pair in pairs)) for i in range(2)),
                percent(mean_defined(abs(a - b) for a, b in both)), len(both), percent(mean_defined(b.review for _, b in pairs)),
            ])
    return ["## Aggregated administrations", "",
            f"Each of {ADMINISTRATIONS:,} synthetic administrations draws one answer per question and does not represent a respondent. The held-out corpus draws from Q3, Q5, Q6, Q11, Q13 and Q14; the contrast set draws from every question it contains, all but Q7. Draws use seed {ADMINISTRATION_SEED}, one random stream per source, questions in numeric order and answer candidates sorted by id. Each applicable domain uses unit weights. No minimum reporting coverage was set. D = 0 cells are shares of administrations; allocation differences are percentage points, restricted to administrations where both reports have D > 0.", "",
            *table(["Source", "Domain", "Items (N)", "D = 0: rater 1 / classifier", "Mean coverage %: rater 1 / classifier", "Mean A allocation % (D > 0): rater 1 / classifier", "Mean absolute A-allocation difference", "Administrations with both D > 0", "Mean review %"], summary),
            "### Example administration", "", "The first contrast-set administration:", "",
            *table(["Domain", "Report", "N", "A-only", "B-only", "Mixed", "Indeterminate (neither / insufficient)", "Invalid", "Review", "Missing", "D", "A allocation", "B allocation", "Indeterminate %", "Invalid %", "Coverage %"], example)]


def report_lines(answers, records, batches, labels, checks) -> list[str]:
    return [
        "# Reference-pattern demonstration", "",
        f"This demonstration applies Appendix C's provisional templates (Table C2) and registry (Table C3), unchanged, not piloted and not frozen, to characteristic labels, with the codebook operationalization departures listed below. The reference is `{ANNOTATOR_MODEL}`'s labels, with `{SECOND_RATER_MODEL}` as comparison; the classifier is the dev model at its calibration thresholds. No clarification probes, relevance or sufficiency component exist, so every answer is treated as relevant and sufficient, every record has prompt count 0, and no record is Invalid or insufficient. Allocation percentages summarise template matches in synthetic answers and do not estimate diagnostic probability.", "",
        *templates_section(labels), *registry_section(answers), *checks,
        *contrast_section(answers, batches), *agreement_section(records),
        *template_section(answers), *confusion_section(records),
        *familiarity_section(records), *indeterminate_section(records),
        *distress_section(answers), *administration_section(answers, records),
        "## Limits", "",
        "- Answers and both raters are language models, not participants or independently adjudicated human annotations. Agreement is with rater 1, not clinical ground truth.",
        "- The registry remains provisional: it was applied without piloting, review or freezing. Calibration scores are optimistic; contrast answers to training questions are new answers, not new dilemmas.",
        "- Conjunctions are answer-level: same-person and same-time links, coherent scope and own-action scope are not verified by binary characteristic labels. The A2, B2 and B3 operationalization departures are stated in the Templates table.",
        "- There are no probes, relevance or sufficiency components. Invalid and insufficient outcomes are supported by the aggregation engine, not detected in these answers. Review reflects threshold sensitivity, not a validated clinical uncertainty rule.",
        "- Contrast-set rates reflect the targeted design and excluded declines, not prevalence in any population. Synthetic administrations mix independent answers and are not respondents.",
        "- Allocation percentages are not probabilities of ASD or ASPD, and this demonstration cannot establish clinical differentiation.", "",
    ]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("referenceDemonstration.py needs a CUDA GPU; torch.cuda.is_available() is False.")
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(f"{PREDICTIONS_PATH} not found; run src/evaluateClassifier.py first.")
    checks = engine_checks()
    answers = held_out_answers()
    batches = contrast_batches()
    rows = contrast_rows(batches)
    labels = load_labels()
    label_keys = [label.machine_key for label in labels]
    thresholds = json.loads(DEV_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    probabilities = predict_splits(DEV_MODEL_DIR, label_keys, {"contrast": rows})["contrast"]
    for row, values in zip(rows, probabilities.tolist(), strict=True):
        probs = dict(zip(label_keys, values, strict=True))
        answers.append(make_answer(row, [key for key in label_keys if probs[key] >= thresholds[key]], probs))
    records = outcome_records(answers, thresholds)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines(answers, records, batches, labels, checks)), encoding="utf-8")
    contrast = [r for r in records if r.source == "contrast"]
    test = [r for r in records if r.source == "test"]
    a, b, r = agreement(contrast, "classifier_outcome"), agreement(contrast, "rater_outcome"), review_share(contrast)
    t, u = agreement(test, "classifier_outcome"), agreement(test, "rater_outcome")
    print(f"Contrast set: classifier agreement {a:.3f}, rater 2 {b:.3f}, review {r:.3f}; test: classifier {t:.3f}, rater 2 {u:.3f}. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
