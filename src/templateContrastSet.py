"""Build template-targeted briefs, generate answers, and annotate them.
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from annotateResponses import (
    ANNOTATOR_REASONING_EFFORT, SECOND_RATER_REASONING_EFFORT,
    annotate_pending, completion_cost, load_template,
)
from buildClassifierData import (
    ANNOTATOR_MODEL, SECOND_RATER_MODEL, annotation_path,
    batch_rows, load_labels, write_jsonl,
)
from evaluateClassifier import question_number
from generatePrompts import (
    GENERATION_CONSTRAINTS, REQUIRES, RESPONSES_PER_PROMPT,
    all_style_pairs, assign_styles, load_prompt_data, render_prompt,
)
from generateResponses import (
    GENERATION_MODELS, REQUEST_TIMEOUT_SECONDS, ROOT_DIR, generate_pending,
)
from referencePatterns import DOMAINS, TEMPLATES


TEMPLATE_PROMPTS_PATH = ROOT_DIR / "data" / "reference_patterns" / "generation_prompts.jsonl"
TEMPLATE_RAW_DIR = ROOT_DIR / "data" / "reference_patterns" / "responses"
TEMPLATE_ANNOTATIONS_ROOT = ROOT_DIR / "data" / "reference_patterns" / "annotations"
BATCHES_PER_BRIEF = 1
SEED = 1

BriefSpec = tuple[str, frozenset[str], frozenset[str]]
BRIEF_SPECS: dict[str, tuple[BriefSpec, ...]] = {
    "A1": (
        ("positive", frozenset({"ce_mentalizing_difficulty", "ce_understood_harm", "ae_welfare_concern"}), frozenset({"ae_explicit_disregard"})),
        ("near_miss", frozenset({"ce_mental_state_attribution", "ce_contextual_explanation", "ce_understood_harm", "ae_welfare_concern"}), frozenset({"ce_mentalizing_difficulty", "ae_explicit_disregard"})),
    ),
    "A2": (
        ("positive", frozenset({"sys_consistent_procedure", "ce_understood_harm", "ae_welfare_concern"}), frozenset({"ae_explicit_disregard"})),
        ("near_miss", frozenset({"sys_rule_justification", "ce_understood_harm", "ae_welfare_concern"}), frozenset({"sys_consistent_procedure", "ae_explicit_disregard"})),
    ),
    "A3": (
        ("positive", frozenset({"sys_consistent_procedure", "ce_understood_harm", "ae_welfare_concern", "pda_harm_regret", "pda_responsibility"}), frozenset({"ae_explicit_disregard", "pda_explicit_no_harm_regret"})),
        ("positive", frozenset({"sys_consistent_procedure", "ce_understood_harm", "ae_welfare_concern", "pda_harm_regret", "pda_repair_intention"}), frozenset({"ae_explicit_disregard", "pda_explicit_no_harm_regret"})),
        ("near_miss", frozenset({"ce_understood_harm", "ae_welfare_concern", "pda_harm_regret", "pda_responsibility"}), frozenset({"sys_consistent_procedure", "ae_explicit_disregard"})),
        ("near_miss", frozenset({"ce_understood_harm", "ae_welfare_concern", "pda_harm_regret", "pda_repair_intention"}), frozenset({"sys_consistent_procedure", "ae_explicit_disregard"})),
    ),
    "B1": (
        ("positive", frozenset({"ce_understood_harm", "ae_explicit_disregard"}), frozenset({"ae_welfare_concern"})),
        ("near_miss", frozenset({"ce_understood_harm", "sys_consequence_reasoning"}), frozenset({"ae_explicit_disregard", "ae_welfare_concern"})),
    ),
    "B2": (
        ("positive", frozenset({"ce_mental_state_attribution", "ce_exploitative_use", "ce_understood_harm", "ae_explicit_disregard"}), frozenset({"ae_welfare_concern"})),
    ) * 2 + (
        ("near_miss", frozenset({"ce_mental_state_attribution", "ce_contextual_explanation", "ce_understood_harm", "sys_consequence_reasoning"}), frozenset({"ce_exploitative_use", "ae_explicit_disregard"})),
    ) * 2,
    "B3": (
        ("positive", frozenset({"sys_instrumental_norm_use", "ce_understood_harm", "ae_explicit_disregard"}), frozenset({"ae_welfare_concern"})),
        ("near_miss", frozenset({"sys_instrumental_norm_use", "ce_understood_harm", "ae_welfare_concern"}), frozenset({"ae_explicit_disregard"})),
    ),
    "B4": (
        ("positive", frozenset({"ce_understood_harm", "pda_explicit_no_harm_regret", "pda_self_consequence_regret", "ae_explicit_disregard"}), frozenset({"ae_welfare_concern", "pda_harm_regret"})),
    ) * 2 + (
        ("near_miss", frozenset({"ce_understood_harm", "pda_harm_regret", "pda_self_consequence_regret"}), frozenset({"pda_explicit_no_harm_regret", "ae_explicit_disregard"})),
    ) * 2,
}
MIXED_SPEC: BriefSpec = (
    "mixed", frozenset({"sys_consistent_procedure", "ce_understood_harm", "ae_welfare_concern", "ae_explicit_disregard"}), frozenset(),
)
SCOPE_BULLET = "- Apply the required traits within a consistent scope: the same focal\n  person, relevant emotion, and time frame."
MIXED_SCOPE_BULLET = "- Acknowledge the harm to two different identified people. Direct concern\n  for welfare at one of them and explicit welfare disregard at the other,\n  and keep each person's traits within one time frame."
if SCOPE_BULLET not in GENERATION_CONSTRAINTS:
    raise ValueError("Generation constraints no longer contain the scope bullet.")
MIXED_CONSTRAINTS = GENERATION_CONSTRAINTS.replace(SCOPE_BULLET, MIXED_SCOPE_BULLET)


def eligible_questions(template_id: str) -> frozenset[str]:
    return frozenset().union(*(
        domain.questions for domain in DOMAINS
        if template_id in domain.a_templates + domain.b_templates
    ))


def build_briefs() -> list[dict[str, Any]]:
    labels, questions, approaches, tones = load_prompt_data()
    label_list = list(labels.values())
    by_key = {label.machine_key: label for label in label_list}
    pairs = all_style_pairs(approaches, tones)
    template = Environment(
        loader=FileSystemLoader(ROOT_DIR / "materials" / "prompts"), autoescape=False,
    ).get_template("generation.jinja2")
    concern = next(domain for domain in DOMAINS if domain.id == "concern")
    records = []
    for question_id in sorted(questions, key=question_number):
        specs = [
            (template_id, spec) for template_id in TEMPLATES
            if question_id in eligible_questions(template_id)
            for spec in BRIEF_SPECS[template_id]
        ]
        if question_id in concern.questions:
            specs.append(("A2+B1", MIXED_SPEC))
        required_sets = []
        for template_id, (_, required, forbidden) in specs:
            if required & forbidden:
                raise ValueError(f"{template_id}: required and forbidden keys overlap")
            for key in required:
                if REQUIRES.get(key, set()) - required:
                    raise ValueError(f"{template_id}: {key} lacks its prerequisite")
            required_sets.append([by_key[key] for key in sorted(required)])
        styles = assign_styles(
            label_list, required_sets, pairs, scenario_id=f"template:{question_id}", seed=SEED,
        )
        for index, ((template_id, spec), required_labels, (approach, tone)) in enumerate(
            zip(specs, required_sets, styles, strict=True), start=1,
        ):
            case, required, forbidden = spec
            records.append({
                "prompt_id": f"{question_id}_T{index:03d}",
                "question_id": question_id, "approach_id": approach.id, "tone_id": tone.id,
                "response_count": RESPONSES_PER_PROMPT,
                "intended_label_keys": sorted(required), "forbidden_label_keys": sorted(forbidden),
                "case": case, "template_id": template_id,
                "prompt": render_prompt(
                    template, questions[question_id], approach, tone, required_labels,
                    [by_key[key] for key in sorted(forbidden)],
                    MIXED_CONSTRAINTS if case == "mixed" else GENERATION_CONSTRAINTS,
                ),
            })
    return records


def contrast_batches() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for model in GENERATION_MODELS
        for path in sorted((TEMPLATE_RAW_DIR / model).glob("*.json"))
    ]


def contrast_rows(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [data for data in batches if data["status"] == "ok"]
    if not ok:
        raise SystemExit("No contrast-set batches; run src/templateContrastSet.py first.")
    missing = sum(
        not annotation_path(data, annotator, root=TEMPLATE_ANNOTATIONS_ROOT).exists()
        for data in ok for annotator in (ANNOTATOR_MODEL, SECOND_RATER_MODEL)
    )
    if missing:
        raise SystemExit(f"{missing} contrast-set annotations missing; run src/templateContrastSet.py until 0 remaining.")
    rows = []
    for data in ok:
        annotations, second = [
            json.loads(annotation_path(data, annotator, root=TEMPLATE_ANNOTATIONS_ROOT).read_text(encoding="utf-8"))["annotations"]
            for annotator in (ANNOTATOR_MODEL, SECOND_RATER_MODEL)
        ]
        rows.extend(
            row | {"case": data["case"], "template_id": data["template_id"]}
            for row in batch_rows(data, "contrast", annotations, second)
        )
    return rows


def main() -> None:
    briefs = build_briefs()
    write_jsonl(TEMPLATE_PROMPTS_PATH, briefs)
    load_dotenv(ROOT_DIR / ".env")
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)
    generate_pending(client, briefs, TEMPLATE_RAW_DIR, BATCHES_PER_BRIEF)
    batches = contrast_batches()
    ok = [data for data in batches if data["status"] == "ok"]
    annotate_pending(
        client, load_template(), load_labels(), load_prompt_data()[1],
        [(ANNOTATOR_MODEL, ANNOTATOR_REASONING_EFFORT, ok),
         (SECOND_RATER_MODEL, SECOND_RATER_REASONING_EFFORT, ok)],
        TEMPLATE_ANNOTATIONS_ROOT,
    )
    for model in GENERATION_MODELS:
        generated = [data for data in batches if data["model"] == model]
        count = sum(data["status"] == "ok" for data in generated)
        declined = sum(data["status"] == "incompatible" for data in generated)
        cost = sum(completion_cost(data) for data in generated)
        print(f"{model}: {count} ok, {declined} declined of {len(briefs)} briefs; generation cost ${cost:.2f}")


if __name__ == "__main__":
    main()
