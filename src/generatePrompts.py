from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import TypeVar

from jinja2 import Environment, FileSystemLoader, Template


# Adjust these to control the generation budget.
MAX_COMBINATIONS_PER_QUESTION = 60
MAX_TRAITS_PER_PROMPT = 3
MAX_FORBIDDEN_PER_PROMPT = 3
# Responses requested per API call. generateResponses.py sends each prompt
# BATCHES_PER_PROMPT times, so a prompt yields a multiple of this.
RESPONSES_PER_PROMPT = 10
SEED = 1
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "main_corpus"
    / "generation_prompts.jsonl"
)


@dataclass(frozen=True)
class Label:
    id: str
    group: str
    label_name: str
    label_description: str
    machine_key: str


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    clarification: str
    reasoning_target: str = ""


@dataclass(frozen=True)
class ResponseApproach:
    id: str
    label: str
    category: str
    instruction: str


@dataclass(frozen=True)
class ResponseTone:
    id: str
    label: str
    valence: str
    activation: str
    instruction: str


Record = TypeVar("Record")


# Required companion labels.
REQUIRES: dict[str, set[str]] = {
    "ce_contextual_explanation": {"ce_mental_state_attribution"},
    "ce_supportive_use": {"ce_mental_state_attribution"},
    "ce_exploitative_use": {"ce_mental_state_attribution"},
}


# Exclude these combinations within the same person/emotion/time frame.
# For this generator, "low distress" means low distress across that frame.
CONFLICTS: list[set[str]] = [
    {"ae_shared_affect", "ae_explicit_nonsharing"},
    {"ae_welfare_concern", "ae_explicit_disregard"},
    {"pda_harm_regret", "pda_explicit_no_harm_regret"},
    {"pd_explicit_low_distress", "pd_harm_distress"},
    {"pd_explicit_low_distress", "pd_decision_conflict"},
    {"pd_explicit_low_distress", "pd_personal_consequence_anxiety"},
]


# Labels an approach or tone instruction tends to elicit.
# Never forbid them in a prompt that uses that approach or tone.
APPROACH_INDUCED_LABELS: dict[str, set[str]] = {
    "R": {"sys_consequence_reasoning", "ce_understood_harm"},
    "A": {"pd_decision_conflict"},
}

TONE_INDUCED_LABELS: dict[str, set[str]] = {
    "T": {
        "pd_harm_distress",
        "pd_decision_conflict",
        "pd_personal_consequence_anxiety",
    },
    "D": {"pd_harm_distress"},
}


# Required labels that contradict an approach or tone instruction.
# Never assign that approach or tone to a prompt requiring them.
APPROACH_INCOMPATIBLE_LABELS: dict[str, set[str]] = {
    "I": {"sys_consistent_procedure"},
    "A": {"sys_consistent_procedure"},
}

TONE_INCOMPATIBLE_LABELS: dict[str, set[str]] = {
    "T": {"pd_explicit_low_distress"},
    "D": {"pd_explicit_low_distress"},
}


GENERATION_CONSTRAINTS = """
Generation constraints:
- Express the required traits in each participant's own everyday words.
  Do not mention label names or machine keys, and do not reuse phrasing
  from the trait descriptions.
- Before writing each response, decide who the participant is and why they
  would reason this way. The required traits should follow from that
  reasoning, not be tacked on to it.
- Apply the required traits within a consistent scope: the same focal
  person, relevant emotion, and time frame.
- Keep every forbidden trait absent from every response.
- Required and forbidden traits take priority if the requested approach or
  tone would contradict them.
- Traits neither required nor forbidden are unspecified, not necessarily
  absent.
- Preserve the scenario's facts. If the required traits cannot fit the
  scenario, or cannot be expressed without a forbidden trait, return ONLY:
  INCOMPATIBLE (instead of the JSON array)
""".strip()


def _read_records(
    path: Path,
    record_type: type[Record],
    key_field: str,
    field_names: dict[str, str] | None = None,
) -> dict[str, Record]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None or key_field not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain a {key_field!r} column"
            )

        records: dict[str, Record] = {}

        for row in reader:
            if None in row:
                raise ValueError(
                    f"{path}, near line {reader.line_num}: "
                    "too many columns; check CSV quoting"
                )

            key = row[key_field]

            if not key:
                raise ValueError(
                    f"{path}, near line {reader.line_num}: "
                    f"empty {key_field!r}"
                )

            if key in records:
                raise ValueError(
                    f"{path} contains duplicate {key_field}: {key}"
                )

            values = {
                field_names.get(name, name) if field_names else name: value
                for name, value in row.items()
            }

            try:
                records[key] = record_type(**values)
            except TypeError as exc:
                raise ValueError(
                    f"{path}, near line {reader.line_num}: {exc}"
                ) from exc

        return records


def load_prompt_data(
    data_dir: Path | str | None = None,
) -> tuple[
    dict[str, Label],
    dict[str, Question],
    dict[str, ResponseApproach],
    dict[str, ResponseTone],
]:
    directory = (
        Path(data_dir)
        if data_dir is not None
        else Path(__file__).resolve().parent.parent / "materials"
    )

    return (
        _read_records(
            directory / "codebook.csv",
            Label,
            "id",
            {
                "labelName": "label_name",
                "labelDescription": "label_description",
                "machineKey": "machine_key",
            },
        ),
        _read_records(
            directory / "scenarios.csv",
            Question,
            "questionID",
            {
                "questionID": "id",
                "questionText": "text",
                "reasoningTarget": "reasoning_target",
            },
        ),
        _read_records(
            directory / "response_approaches.csv",
            ResponseApproach,
            "id",
        ),
        _read_records(
            directory / "response_tones.csv",
            ResponseTone,
            "id",
        ),
    )


def generate_trait_combinations(
    labels: list[Label],
    scenario_id: str,
    *,
    max_combinations: int = 60,
    max_traits: int = 3,
    seed: int = 1,
) -> list[list[Label]]:
    """
    Select compatible positive-trait combinations.

    First include a simple anchor for every label, adding dependencies.
    Then greedily select combinations that cover new positive-label pairs,
    favoring less-used labels when pair coverage is tied.

    max_traits includes dependency labels.
    Unselected labels are unspecified.
    """
    if not 1 <= max_traits <= 4:
        raise ValueError("max_traits must be between 1 and 4.")

    if max_combinations < 1:
        raise ValueError("max_combinations must be positive.")

    by_key = {label.machine_key: label for label in labels}

    if len(by_key) != len(labels):
        raise ValueError("Labels must have unique machine_key values.")

    if any(not key.strip() for key in by_key):
        raise ValueError("Labels must have nonempty machine_key values.")

    if not labels:
        return []

    keys = sorted(by_key)
    rng = random.Random(f"{seed}:{scenario_id}")

    def expand(traits: set[str]) -> frozenset[str]:
        expanded = set(traits)

        while True:
            required: set[str] = set()

            for key in expanded:
                required.update(REQUIRES.get(key, set()))

            missing = required - by_key.keys()
            if missing:
                raise ValueError(
                    f"Missing required labels: {sorted(missing)}"
                )

            if required <= expanded:
                return frozenset(expanded)

            expanded.update(required)

    def valid(traits: frozenset[str]) -> bool:
        return (
            len(traits) <= max_traits
            and not any(conflict <= traits for conflict in CONFLICTS)
        )

    def pairs(traits: frozenset[str]) -> set[tuple[str, str]]:
        return set(combinations(sorted(traits), 2))

    anchors: list[frozenset[str]] = []

    for key in keys:
        anchor = expand({key})

        if not valid(anchor):
            raise ValueError(
                f"{key!r} cannot fit the current constraints. "
                "Check max_traits, REQUIRES, and CONFLICTS."
            )

        if anchor not in anchors:
            anchors.append(anchor)

    if max_combinations < len(anchors):
        raise ValueError(
            f"Need at least {len(anchors)} combinations "
            "to include all positive anchors."
        )

    candidates: set[frozenset[str]] = set()

    for size in range(1, min(max_traits, len(keys)) + 1):
        for raw in combinations(keys, size):
            candidate = expand(set(raw))

            if valid(candidate):
                candidates.add(candidate)

    selected = anchors.copy()

    remaining = sorted(
        candidates - set(selected),
        key=lambda traits: tuple(sorted(traits)),
    )
    rng.shuffle(remaining)

    covered_pairs: set[tuple[str, str]] = set()
    counts = dict.fromkeys(keys, 0)

    for traits in selected:
        covered_pairs.update(pairs(traits))
        for key in traits:
            counts[key] += 1

    pair_cache = {
        traits: pairs(traits)
        for traits in remaining
    }

    def score(traits: frozenset[str]) -> tuple[int, float, int]:
        new_pairs = len(pair_cache[traits] - covered_pairs)
        average_frequency = sum(counts[key] for key in traits) / len(traits)

        return (
            new_pairs,
            -average_frequency,
            -len(traits),
        )

    while remaining and len(selected) < max_combinations:
        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)

        covered_pairs.update(pair_cache[best])

        for key in best:
            counts[key] += 1

    return [
        [by_key[key] for key in sorted(traits)]
        for traits in selected
    ]


def assign_styles(
    labels: list[Label],
    required_sets: list[list[Label]],
    style_pairs: list[tuple[ResponseApproach, ResponseTone]],
    scenario_id: str,
    *,
    seed: int = 1,
) -> list[tuple[ResponseApproach, ResponseTone]]:
    """
    Assign one approach/tone pair to each required label set.

    Pairs are dealt from shuffled decks of all pairs, so each pair is used
    about equally often. A pair whose approach or tone contradicts a
    required label is skipped and stays in the deck for a later set.
    """
    label_keys = {label.machine_key for label in labels}
    unknown = set().union(
        *APPROACH_INCOMPATIBLE_LABELS.values(),
        *TONE_INCOMPATIBLE_LABELS.values(),
    ) - label_keys

    if unknown:
        raise ValueError(
            f"Unknown style-incompatible labels: {sorted(unknown)}"
        )

    def incompatible(style: tuple[ResponseApproach, ResponseTone]) -> set[str]:
        approach, tone = style
        return APPROACH_INCOMPATIBLE_LABELS.get(
            approach.id, set()
        ) | TONE_INCOMPATIBLE_LABELS.get(tone.id, set())

    rng = random.Random(f"styles:{seed}:{scenario_id}")
    deck: list[tuple[ResponseApproach, ResponseTone]] = []
    styles: list[tuple[ResponseApproach, ResponseTone]] = []

    for required_labels in required_sets:
        required = {label.machine_key for label in required_labels}

        if all(incompatible(style) & required for style in style_pairs):
            raise ValueError(
                f"No style pair is compatible with {sorted(required)}."
            )

        while True:
            index = next(
                (
                    position
                    for position in reversed(range(len(deck)))
                    if not incompatible(deck[position]) & required
                ),
                None,
            )

            if index is not None:
                styles.append(deck.pop(index))
                break

            fresh = style_pairs.copy()
            rng.shuffle(fresh)
            deck = fresh + deck

    return styles


def select_forbidden_labels(
    labels: list[Label],
    required_sets: list[list[Label]],
    styles: list[tuple[ResponseApproach, ResponseTone]],
    scenario_id: str,
    *,
    max_forbidden: int = 3,
    seed: int = 1,
) -> list[list[Label]]:
    """
    Select hard-negative labels that each prompt's responses must omit.

    Candidates exclude required labels and labels the prompt's approach or
    tone tends to elicit. Prefer CONFLICTS partners of required labels, then
    labels sharing a group with a required label, then any other label.
    Within a tier, prefer labels forbidden least often for this scenario.

    max_forbidden counts selected labels. Forbidding a label also forbids
    every label that REQUIRES it, so a set can exceed max_forbidden.
    """
    if max_forbidden < 0:
        raise ValueError("max_forbidden must be non-negative.")

    by_key = {label.machine_key: label for label in labels}
    keys = sorted(by_key)

    unknown = set().union(
        *APPROACH_INDUCED_LABELS.values(),
        *TONE_INDUCED_LABELS.values(),
    ) - by_key.keys()

    if unknown:
        raise ValueError(f"Unknown style-induced labels: {sorted(unknown)}")

    dependents: dict[str, set[str]] = {key: set() for key in keys}

    for dependent, prerequisites in REQUIRES.items():
        if dependent not in by_key:
            continue

        for prerequisite in prerequisites:
            if prerequisite in by_key:
                dependents[prerequisite].add(dependent)

    units: dict[str, frozenset[str]] = {}

    for key in keys:
        unit = {key}
        pending = [key]

        while pending:
            for dependent in dependents[pending.pop()]:
                if dependent not in unit:
                    unit.add(dependent)
                    pending.append(dependent)

        units[key] = frozenset(unit)

    rng = random.Random(f"forbidden:{seed}:{scenario_id}")
    order = keys.copy()
    rng.shuffle(order)
    position = {key: index for index, key in enumerate(order)}

    counts = dict.fromkeys(keys, 0)
    forbidden_sets: list[list[Label]] = []

    for required_labels, (approach, tone) in zip(
        required_sets, styles, strict=True
    ):
        required = {label.machine_key for label in required_labels}
        required_groups = {label.group for label in required_labels}
        blocked = (
            required
            | APPROACH_INDUCED_LABELS.get(approach.id, set())
            | TONE_INDUCED_LABELS.get(tone.id, set())
        )

        conflict_partners = {
            key
            for conflict in CONFLICTS
            for key in conflict
            if conflict - {key} <= required
        }
        tiers = {
            key: 0 if key in conflict_partners
            else 1 if by_key[key].group in required_groups
            else 2
            for key in keys
        }

        forbidden: set[str] = set()
        chosen = 0

        while chosen < max_forbidden:
            options = [
                key
                for key in keys
                if key not in forbidden and not units[key] & blocked
            ]

            if not options:
                break

            best = min(
                options,
                key=lambda key: (tiers[key], counts[key], position[key]),
            )
            forbidden |= units[best]
            chosen += 1

            for key in units[best]:
                counts[key] += 1

        forbidden_sets.append([by_key[key] for key in sorted(forbidden)])

    return forbidden_sets


def all_style_pairs(
    response_approaches: dict[str, ResponseApproach],
    response_tones: dict[str, ResponseTone],
) -> list[tuple[ResponseApproach, ResponseTone]]:
    return list(product(
        sorted(response_approaches.values(), key=lambda item: item.id),
        sorted(response_tones.values(), key=lambda item: item.id),
    ))


def render_prompt(
    template: Template, question: Question,
    approach: ResponseApproach, tone: ResponseTone,
    required_labels: list[Label], forbidden_labels: list[Label],
    constraints: str = GENERATION_CONSTRAINTS,
) -> str:
    prompt = template.render(
        questionText=question.text,
        responseCount=RESPONSES_PER_PROMPT,
        clarification=question.clarification,
        reasoningTarget=question.reasoning_target,
        approachInstruction=approach.instruction,
        toneInstruction=tone.instruction,
        requiredLabels=required_labels,
        forbiddenLabels=forbidden_labels,
    )
    return f"{prompt.rstrip()}\n\n{constraints}"


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "materials"

    (
        labels,
        questions,
        response_approaches,
        response_tones,
    ) = load_prompt_data(data_dir)

    if not labels:
        raise ValueError("codebook.csv contains no labels.")

    if not questions:
        raise ValueError("scenarios.csv contains no questions.")

    env = Environment(
        loader=FileSystemLoader(data_dir / "prompts"),
        autoescape=False,
    )
    template = env.get_template("generation.jinja2")

    label_list = list(labels.values())

    style_pairs = all_style_pairs(response_approaches, response_tones)

    if not style_pairs:
        raise ValueError(
            "At least one response approach and tone are required."
        )

    records: list[dict[str, object]] = []

    for question_id, question_data in sorted(questions.items()):
        trait_sets = generate_trait_combinations(
            label_list,
            scenario_id=question_id,
            max_combinations=MAX_COMBINATIONS_PER_QUESTION,
            max_traits=MAX_TRAITS_PER_PROMPT,
            seed=SEED,
        )

        styles = assign_styles(
            label_list,
            trait_sets,
            style_pairs,
            scenario_id=question_id,
            seed=SEED,
        )

        forbidden_sets = select_forbidden_labels(
            label_list,
            trait_sets,
            styles,
            scenario_id=question_id,
            max_forbidden=MAX_FORBIDDEN_PER_PROMPT,
            seed=SEED,
        )

        for index, (required_labels, style, forbidden_labels) in enumerate(
            zip(trait_sets, styles, forbidden_sets, strict=True),
            start=1,
        ):
            approach, tone = style

            prompt = render_prompt(
                template, question_data, approach, tone,
                required_labels, forbidden_labels,
            )

            record = {
                "prompt_id": f"{question_id}_{index:03d}",
                "question_id": question_id,
                "approach_id": approach.id,
                "tone_id": tone.id,
                "response_count": RESPONSES_PER_PROMPT,
                "intended_label_keys": [
                    label.machine_key for label in required_labels
                ],
                "forbidden_label_keys": [
                    label.machine_key for label in forbidden_labels
                ],
                "prompt": prompt,
            }

            records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()