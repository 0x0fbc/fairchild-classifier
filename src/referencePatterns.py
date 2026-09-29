"""Deterministic reference-pattern scoring (Appendix C2–C4, C6).

Inputs are sets of codebook label keys. Conjunctions are checked at the answer
level, so Table C2's same-person and same-time links are not verified.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from buildClassifierData import annotated_label_keys
from evaluateClassifier import THRESHOLD_HUNDREDTHS


@dataclass(frozen=True)
class ReferenceTemplate:
    id: str
    name: str
    all_of: frozenset[str]
    any_of: tuple[frozenset[str], ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ReportDomain:
    id: str
    name: str
    questions: frozenset[str]
    a_templates: tuple[str, ...]
    b_templates: tuple[str, ...]


TEMPLATES = {
    "A1": ReferenceTemplate(
        "A1", "Interpretive difficulty with concern",
        frozenset({"ce_mentalizing_difficulty", "ce_understood_harm", "ae_welfare_concern"}),
    ),
    "A2": ReferenceTemplate(
        "A2", "Systematic principle with acknowledged concern",
        frozenset({"sys_consistent_procedure", "ce_understood_harm", "ae_welfare_concern"}),
        note="RULE is implied: consistent decision procedure is one of RULE's codebook labels.",
    ),
    "A3": ReferenceTemplate(
        "A3", "Reflective harm appraisal",
        frozenset({"sys_consistent_procedure", "ce_understood_harm", "ae_welfare_concern", "pda_harm_regret"}),
        (frozenset({"pda_responsibility", "pda_repair_intention"}),),
    ),
    "B1": ReferenceTemplate(
        "B1", "Understood harm with explicit disregard",
        frozenset({"ce_understood_harm", "ae_explicit_disregard"}),
    ),
    "B2": ReferenceTemplate(
        "B2", "Exploitative use of mental-state knowledge",
        frozenset({"ce_exploitative_use", "ce_understood_harm", "ae_explicit_disregard"}),
        (frozenset({"ce_mental_state_attribution", "ce_contextual_explanation"}),),
        "AGENT_PERSPECTIVE is not coded, so only AFFECTED_PERSPECTIVE (CE1 or CE2) meets the perspective requirement.",
    ),
    "B3": ReferenceTemplate(
        "B3", "Instrumental norm compliance",
        frozenset({"sys_instrumental_norm_use", "ce_understood_harm", "ae_explicit_disregard"}),
        note="SELF_GAIN_PRIORITY and SELF_PROTECTION are not coded; instrumental rule compliance (followed only while it secures an advantage or prevents a cost) is taken to state one of them.",
    ),
    "B4": ReferenceTemplate(
        "B4", "Self-consequence appraisal without harm regret",
        frozenset({"ce_understood_harm", "pda_explicit_no_harm_regret", "pda_self_consequence_regret", "ae_explicit_disregard"}),
    ),
}

DOMAINS = (
    ReportDomain("concern", "Affective domain: concern-related reasoning",
                 frozenset({"Q1", "Q3", "Q4", "Q5", "Q6", "Q8", "Q10", "Q12", "Q14", "Q15"}),
                 ("A1", "A2"), ("B1",)),
    ReportDomain("cognitive_use", "Cognitive domain: use of mental-state information",
                 frozenset({"Q9", "Q10", "Q11", "Q12", "Q13"}), ("A1",), ("B2",)),
    ReportDomain("distress", "Personal distress",
                 frozenset({"Q1", "Q4", "Q5", "Q6", "Q7", "Q8", "Q11", "Q12", "Q15"}), (), ()),
    ReportDomain("rule", "Systemizing-related domain: rule justification",
                 frozenset({"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q9", "Q10", "Q11", "Q12", "Q13"}),
                 ("A2",), ("B3",)),
    ReportDomain("appraisal", "Post-decisional appraisal",
                 frozenset({"Q8", "Q12", "Q15"}), ("A3",), ("B4",)),
)

A_ONLY, B_ONLY, MIXED = "A-only", "B-only", "Mixed"
NEITHER = "Indeterminate: neither pattern"
INSUFFICIENT = "Indeterminate: insufficient reasoning"
INVALID, REVIEW, MISSING = "Invalid", "Processing review", "Missing or declined"
STATUSES = (A_ONLY, B_ONLY, MIXED, NEITHER, INSUFFICIENT, INVALID, REVIEW, MISSING)
OUTCOMES = (A_ONLY, B_ONLY, MIXED, NEITHER)
REVIEW_MARGIN = 0.10


def with_prerequisites(keys: Iterable[str]) -> frozenset[str]:
    return frozenset(annotated_label_keys(keys))


def template_met(template: ReferenceTemplate, present: frozenset[str]) -> bool:
    return template.all_of <= present and all(group & present for group in template.any_of)


def domain_outcome(domain: ReportDomain, present: frozenset[str]) -> str:
    if not domain.a_templates and not domain.b_templates:
        raise ValueError(f"{domain.id} has no A/B templates")
    support_a = any(template_met(TEMPLATES[t], present) for t in domain.a_templates)
    support_b = any(template_met(TEMPLATES[t], present) for t in domain.b_templates)
    if support_a and support_b:
        return MIXED
    if support_a:
        return A_ONLY
    return B_ONLY if support_b else NEITHER


def shifted_present(
    probabilities: dict[str, float], thresholds: dict[str, float], shift: float,
) -> frozenset[str]:
    lower, upper = THRESHOLD_HUNDREDTHS[0] / 100, THRESHOLD_HUNDREDTHS[-1] / 100
    return with_prerequisites(
        key for key, threshold in thresholds.items()
        if probabilities[key] >= min(max(round(threshold + shift, 2), lower), upper)
    )


def gated_outcome(
    domain: ReportDomain, predicted: frozenset[str],
    probabilities: dict[str, float], thresholds: dict[str, float],
) -> str:
    base = domain_outcome(domain, predicted)
    lenient = domain_outcome(domain, shifted_present(probabilities, thresholds, -REVIEW_MARGIN))
    strict = domain_outcome(domain, shifted_present(probabilities, thresholds, REVIEW_MARGIN))
    return base if lenient == strict else REVIEW


@dataclass(frozen=True)
class Aggregate:
    counts: dict[str, int]
    n: int
    d: int
    a_allocation: float | None
    b_allocation: float | None
    indeterminate: float | None
    invalid: float | None
    coverage: float | None
    review: float | None
    missing: float | None


def aggregate(statuses: Sequence[str]) -> Aggregate:
    counts = dict.fromkeys(STATUSES, 0)
    for status in statuses:
        if status not in counts:
            raise ValueError(f"Unknown reference status: {status}")
        counts[status] += 1
    n = len(statuses)
    d = counts[A_ONLY] + counts[B_ONLY] + counts[MIXED]
    return Aggregate(
        counts, n, d,
        100 * (counts[A_ONLY] + 0.5 * counts[MIXED]) / d if d else None,
        100 * (counts[B_ONLY] + 0.5 * counts[MIXED]) / d if d else None,
        100 * (counts[NEITHER] + counts[INSUFFICIENT]) / n if n else None,
        100 * counts[INVALID] / n if n else None,
        100 * d / n if n else None,
        100 * counts[REVIEW] / n if n else None,
        100 * counts[MISSING] / n if n else None,
    )
