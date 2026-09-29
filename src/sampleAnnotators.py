"""
Choose the annotator by comparing candidate runs on a fixed sample.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from annotateResponses import (
    completion_cost,
    design_agreement,
    load_template,
    run_annotation,
)
from buildClassifierData import (
    annotated_label_keys,
    load_labels,
    source_batches,
)
from generatePrompts import load_prompt_data
from generateResponses import REQUEST_TIMEOUT_SECONDS, ROOT_DIR, write_json


# (run name, model, reasoning effort); None leaves the model's default.
# The first run is the reference; the second repeats it to measure how
# well the reference agrees with itself. Comments give Venice's USD price
# per million input/output tokens.
ANNOTATOR_RUNS: list[tuple[str, str, str | None]] = [
    ("claude-opus-5-5", "claude-opus-5-5", "medium"),  # $4.80/$24
    ("claude-opus-5-5-repeat", "claude-opus-5-5", "medium"),
    ("claude-opus-5-5-low", "claude-opus-5-5", "low"),
    ("claude-sonnet-5", "claude-sonnet-5", None),  # $3/$15
    ("openai-gpt-6-sol", "openai-gpt-6-sol", None),  # $2.50/$12.50
    ("gemini-3-1-pro-preview", "gemini-3-1-pro-preview", None),  # $2.50/$15
    ("grok-4-7", "grok-4-7", None),  # $2.27/$6.80
    ("deepseek-v4-pro-0813", "deepseek-v4-pro-0813", None),  # $1.65/$4.95
    ("gemini-3-8-flash", "gemini-3-8-flash", None),  # $0.94/$4.69
    ("openai-gpt-6-luna", "openai-gpt-6-luna", None),  # $0.125/$0.625
    ("qwen-3-8-flash", "qwen-3-8-flash", None),  # $0.14/$0.49
    ("deepseek-v4-flash", "deepseek-v4-flash", None),  # $0.138/$0.275
    ("z-ai-glm-5-3-flash", "z-ai-glm-5-3-flash", None),  # $0.15/$0.50
    ("google-gemma-4-31b-it", "google-gemma-4-31b-it", None),  # $0.12/$0.36
]
SAMPLE_OPUS_BATCHES = 24
SAMPLE_SEED = 1
CONCURRENT_REQUESTS = 8
# Requests in the full annotation run: ~1,763 Opus + ~1,800 Sonnet + 30
# samples.
FULL_RUN_REQUESTS = 3600
# Selection rule.
MIN_RELATIVE_F1 = 0.95
MAX_DESIGN_AGREEMENT_DROP = 0.02
MIN_OK_BATCHES = 28
MIN_SUCCESS_RATE = 0.9
MAX_MEDIAN_SECONDS = 180
# A candidate run whose median over at least this many timed attempts
# exceeds MAX_MEDIAN_SECONDS gets no further requests, since it is too slow
# to qualify.
SPEED_CHECK_ATTEMPTS = 8
MIN_REFERENCE_POSITIVES = 5
SAMPLE_DIR = ROOT_DIR / "data" / "pilots" / "annotators"
REQUEST_LOG_PATH = SAMPLE_DIR / "requests.jsonl"
REPORT_PATH = ROOT_DIR / "results" / "pilots" / "annotators.md"
SELECTION_PATH = SAMPLE_DIR / "selection.json"


@dataclass
class RunScores:
    run: str
    model: str
    effort: str | None
    batches: int
    success_rate: float
    median_seconds: float
    cost_per_request: float
    labels_per_response: float
    intended_present: float
    forbidden_absent: float
    label_f1: dict[str, float] = field(default_factory=dict)
    macro_f1: float = math.nan
    relative_f1: float = math.nan
    pair_agreement: float = math.nan
    # Qualification criteria the run fails; None for the reference and repeat.
    failures: list[str] | None = None


def batch_id(data: dict[str, Any]) -> str:
    return f"{data['model']}/{data['prompt_id']}_b{data['batch']}"


def sample_batches() -> list[dict[str, Any]]:
    opus = [
        data
        for data, _ in source_batches()
        if data["model"] == "claude-opus-5-5"
    ]
    random.Random(SAMPLE_SEED).shuffle(opus)
    sonnet_dir = ROOT_DIR / "data" / "pilots" / "generators" / "claude-sonnet-5"
    sonnet = [
        data
        for path in sorted(sonnet_dir.glob("*_b1.json"))
        if (data := json.loads(path.read_text(encoding="utf-8")))["status"]
        == "ok"
    ]
    return opus[:SAMPLE_OPUS_BATCHES] + sonnet


def sample_path(run: str, data: dict[str, Any]) -> Path:
    return (
        SAMPLE_DIR
        / run
        / data["model"]
        / f"{data['prompt_id']}_b{data['batch']}.json"
    )


def present_keys(
    annotations: dict[str, list[dict[str, str]]],
) -> dict[str, list[set[str]]]:
    return {
        key: [set(annotated_label_keys(annotation)) for annotation in batch]
        for key, batch in annotations.items()
    }


def score_run(
    run: str,
    model: str,
    effort: str | None,
    annotations: dict[str, list[dict[str, str]]],
    attempts: list[dict[str, Any]],
    reference: dict[str, list[dict[str, str]]],
    batches_by_id: dict[str, dict[str, Any]],
    eligible: list[str],
    label_count: int,
) -> RunScores:
    seconds = [
        attempt["elapsed_seconds"]
        for attempt in attempts
        if attempt["elapsed_seconds"] is not None
    ]
    cost = sum(attempt["cost_usd"] for attempt in attempts)
    present = present_keys(annotations)
    responses = [keys for batch in present.values() for keys in batch]
    common = [key for key in annotations if key in reference]
    intended_present, _, forbidden_absent, _ = design_agreement(
        [(batches_by_id[key], annotations[key]) for key in common]
    )
    scores = RunScores(
        run=run,
        model=model,
        effort=effort,
        batches=len(annotations),
        success_rate=(
            sum(attempt["status"] == "ok" for attempt in attempts)
            / len(attempts)
            if attempts
            else math.nan
        ),
        median_seconds=statistics.median(seconds) if seconds else math.nan,
        cost_per_request=cost / len(annotations) if annotations else math.nan,
        labels_per_response=(
            statistics.mean(map(len, responses)) if responses else math.nan
        ),
        intended_present=intended_present,
        forbidden_absent=forbidden_absent,
    )

    if annotations is reference:
        return scores

    reference_present = present_keys({key: reference[key] for key in common})
    counts = {key: Counter() for key in eligible}
    matches = pairs = 0

    for key in common:
        for keys, truth in zip(
            present[key], reference_present[key], strict=True
        ):
            for label in eligible:
                outcome = (label in keys, label in truth)
                counts[label][outcome] += 1

            matches += label_count - len(keys ^ truth)
            pairs += label_count

    for label, outcome_counts in counts.items():
        true_positives = outcome_counts[True, True]
        errors = outcome_counts[True, False] + outcome_counts[False, True]
        denominator = 2 * true_positives + errors
        scores.label_f1[label] = (
            2 * true_positives / denominator if denominator else math.nan
        )

    defined = [f1 for f1 in scores.label_f1.values() if not math.isnan(f1)]
    scores.macro_f1 = statistics.mean(defined) if defined else math.nan
    scores.pair_agreement = matches / pairs if pairs else math.nan
    return scores


def qualification_failures(
    scores: RunScores,
    reference: RunScores,
    reference_intended_present: float,
    reference_forbidden_absent: float,
) -> list[str]:
    checks = [
        ("batches", scores.batches >= MIN_OK_BATCHES),
        ("success", scores.success_rate >= MIN_SUCCESS_RATE),
        ("speed", scores.median_seconds <= MAX_MEDIAN_SECONDS),
        ("F1", scores.relative_f1 >= MIN_RELATIVE_F1),
        (
            "design",
            scores.intended_present
            >= reference_intended_present - MAX_DESIGN_AGREEMENT_DROP
            and scores.forbidden_absent
            >= reference_forbidden_absent - MAX_DESIGN_AGREEMENT_DROP,
        ),
        ("cost", scores.cost_per_request < reference.cost_per_request),
    ]
    return [name for name, passed in checks if not passed]


def number(value: float, digits: int = 3) -> str:
    return "–" if math.isnan(value) else f"{value:.{digits}f}"


def effort_name(effort: str | None) -> str:
    return "default" if effort is None else effort


def table_row(scores: RunScores) -> str:
    if scores.failures is None:
        qualifies = "–"
    elif scores.failures:
        qualifies = "no: " + ", ".join(scores.failures)
    else:
        qualifies = "yes"

    projected = scores.cost_per_request * FULL_RUN_REQUESTS
    cells = [
        f"`{scores.run}`",
        f"`{scores.model}`",
        effort_name(scores.effort),
        str(scores.batches),
        number(scores.success_rate, 2),
        number(scores.median_seconds, 0),
        number(scores.cost_per_request, 4),
        "–" if math.isnan(projected) else f"${projected:.0f}",
        number(scores.labels_per_response, 2),
        number(scores.macro_f1),
        number(scores.relative_f1),
        number(scores.pair_agreement),
        number(scores.intended_present),
        number(scores.forbidden_absent),
        qualifies,
    ]
    return "| " + " | ".join(cells) + " |"


def load_attempts() -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = {
        run: [] for run, _, _ in ANNOTATOR_RUNS
    }

    if REQUEST_LOG_PATH.exists():
        with REQUEST_LOG_PATH.open(encoding="utf-8") as log_file:
            for attempt in map(json.loads, log_file):
                attempts.setdefault(attempt["run"], []).append(attempt)

    return attempts


def too_slow_runs(attempts: dict[str, list[dict[str, Any]]]) -> list[str]:
    slow: list[str] = []

    for run, _, _ in ANNOTATOR_RUNS[2:]:
        seconds = [
            attempt["elapsed_seconds"]
            for attempt in attempts[run]
            if attempt["elapsed_seconds"] is not None
        ]

        if (
            len(seconds) >= SPEED_CHECK_ATTEMPTS
            and statistics.median(seconds) > MAX_MEDIAN_SECONDS
        ):
            slow.append(run)

    return slow


def write_report() -> str:
    batches = sample_batches()
    batches_by_id = {batch_id(data): data for data in batches}
    label_keys = [label.machine_key for label in load_labels()]
    attempts = load_attempts()

    annotations: dict[str, dict[str, list[dict[str, str]]]] = {}

    for run, _, _ in ANNOTATOR_RUNS:
        annotations[run] = {}

        for data in batches:
            path = sample_path(run, data)

            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                annotations[run][batch_id(data)] = saved["annotations"]

    reference_run, repeat_run = ANNOTATOR_RUNS[0][0], ANNOTATOR_RUNS[1][0]
    reference = annotations[reference_run]
    reference_counts = Counter(
        label
        for batch in present_keys(reference).values()
        for keys in batch
        for label in keys
    )
    eligible = [
        label
        for label in label_keys
        if reference_counts[label] >= MIN_REFERENCE_POSITIVES
    ]
    scores = [
        score_run(
            run,
            model,
            effort,
            annotations[run],
            attempts[run],
            reference,
            batches_by_id,
            eligible,
            len(label_keys),
        )
        for run, model, effort in ANNOTATOR_RUNS
    ]
    reference_scores, repeat_scores = scores[0], scores[1]

    for run_scores in scores[1:]:
        run_scores.relative_f1 = (
            run_scores.macro_f1 / repeat_scores.macro_f1
            if repeat_scores.macro_f1
            else math.nan
        )

    for run_scores in scores[2:]:
        common = [
            (batches_by_id[key], reference[key])
            for key in annotations[run_scores.run]
            if key in reference
        ]
        reference_intended, _, reference_forbidden, _ = design_agreement(
            common
        )
        run_scores.failures = qualification_failures(
            run_scores,
            reference_scores,
            reference_intended,
            reference_forbidden,
        )

    answer_count = sum(len(data["responses"]) for data in batches)
    opus_count = sum(data["model"] == "claude-opus-5-5" for data in batches)
    reference_effort = effort_name(ANNOTATOR_RUNS[0][2])
    lines = [
        "# Annotator comparison",
        "",
        "**Preliminary annotator-selection comparison.** Pilot annotations "
        "are in `data/pilots/annotators/` (repository-root-relative); final "
        "main-study and cross-generator annotations are separately retained "
        "in `data/main_corpus/annotations/`. This 300-answer pilot is not the "
        "main study's 300-answer cross-generator evaluation. See the "
        "[repository guide](../../README.md) for the current manuscript and "
        "reproduction workflow.",
        "",
        f"Generated by `src/sampleAnnotators.py`. Every run annotated the "
        f"same {len(batches)} batches ({answer_count} answers): {opus_count} "
        f"randomly chosen `claude-opus-5-5` raw batches and "
        f"{len(batches) - opus_count} `claude-sonnet-5` sample batches. "
        f"Agreement is measured against the reference run `{reference_run}` "
        f"(`{ANNOTATOR_RUNS[0][1]}`, reasoning effort {reference_effort}); "
        f"its repeat `{repeat_run}` shows how well the reference agrees with "
        f"itself, with macro F1 {number(repeat_scores.macro_f1)}. Macro F1 "
        f"averages per-label F1, with the reference as truth, over the "
        f"{len(eligible)} labels the reference marks present in at least "
        f"{MIN_REFERENCE_POSITIVES} answers; relative F1 divides it by the "
        f"repeat's. Pair agreement covers all {len(label_keys)} labels. "
        f"Intended present and forbidden absent compare a run with the prompt "
        f"design. Agreement and design shares cover the batches a run shares "
        f"with the reference.",
        "",
        f"Success: ok attempts as a share of logged attempts. Median s: "
        f"median seconds per attempt. $/request: the cost of all logged "
        f"attempts, failures included, divided by saved batches. Projected "
        f"full run: $/request x {FULL_RUN_REQUESTS} requests.",
        "",
        f"A run qualifies with at least {MIN_OK_BATCHES} batches, a success "
        f"rate of at least {MIN_SUCCESS_RATE}, a median of at most "
        f"{MAX_MEDIAN_SECONDS} s, relative F1 of at least {MIN_RELATIVE_F1}, "
        f"both design shares no more than {MAX_DESIGN_AGREEMENT_DROP} below "
        f"the reference's on the same batches, and a lower cost per request "
        f"than the reference. The cheapest qualifying run is selected, higher "
        f"macro F1 breaking ties; if none qualifies, the reference is. A "
        f"candidate run whose median exceeds {MAX_MEDIAN_SECONDS} s after at "
        f"least {SPEED_CHECK_ATTEMPTS} attempts gets no further requests, so "
        f"its batches stay incomplete.",
        "",
        "| Run | Model | Effort | Batches | Success | Median s | $/request "
        "| Projected full run | Labels/response | Macro F1 | Relative F1 "
        "| Pair agreement | Intended present | Forbidden absent | Qualifies |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *map(table_row, scores),
        "",
        "## Per-label F1 against the reference",
        "",
        "| Label | Reference positives | "
        + " | ".join(f"`{run_scores.run}`" for run_scores in scores[1:])
        + " |",
        "|---|---|" + "---|" * (len(scores) - 1),
    ]

    for label in eligible:
        cells = [
            f"`{label}`",
            str(reference_counts[label]),
            *(number(run_scores.label_f1[label]) for run_scores in scores[1:]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    if len(reference) < len(batches) or len(annotations[repeat_run]) < len(
        batches
    ):
        outcome = (
            f"Selection pending: the reference and repeat runs need all "
            f"{len(batches)} batches; rerun src/sampleAnnotators.py."
        )
    else:
        qualifying = [
            run_scores for run_scores in scores[2:] if not run_scores.failures
        ]
        selected = (
            min(
                qualifying,
                key=lambda run_scores: (
                    run_scores.cost_per_request,
                    -run_scores.macro_f1,
                ),
            )
            if qualifying
            else reference_scores
        )
        projected = selected.cost_per_request * FULL_RUN_REQUESTS
        is_reference = selected is reference_scores
        write_json(
            SELECTION_PATH,
            {
                "run": selected.run,
                "model": selected.model,
                "reasoning_effort": selected.effort,
                "cost_per_request": round(selected.cost_per_request, 6),
                "projected_full_run_usd": round(projected, 2),
                "macro_f1": (
                    None if is_reference else round(selected.macro_f1, 4)
                ),
                "relative_f1": (
                    None if is_reference else round(selected.relative_f1, 4)
                ),
            },
        )
        outcome = (
            f"Selected annotator: {selected.run} ({selected.model}, reasoning "
            f"effort {effort_name(selected.effort)}); "
            f"${selected.cost_per_request:.4f} per request, projected "
            f"${projected:.0f} for {FULL_RUN_REQUESTS} requests."
        )

    lines.append(outcome)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)
    template = load_template()
    labels = load_labels()
    questions = load_prompt_data()[1]
    batches = sample_batches()
    attempts = load_attempts()
    too_slow = set(too_slow_runs(attempts))
    jobs = [
        (run, model, effort, data)
        for data in batches
        for run, model, effort in ANNOTATOR_RUNS
        if run not in too_slow and not sample_path(run, data).exists()
    ]
    total = len(batches) * len(ANNOTATOR_RUNS)
    saved = sum(
        sample_path(run, data).exists()
        for data in batches
        for run, _, _ in ANNOTATOR_RUNS
    )
    print(
        f"{saved} of {total} sample annotations already saved "
        f"({len(batches)} batches x {len(ANNOTATOR_RUNS)} runs); requesting "
        f"{len(jobs)} now, {CONCURRENT_REQUESTS} at a time."
    )

    if too_slow:
        print(
            f"No further requests for runs too slow to qualify (median over "
            f"{MAX_MEDIAN_SECONDS} s after at least {SPEED_CHECK_ATTEMPTS} "
            f"attempts): {', '.join(sorted(too_slow))}."
        )

    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    with REQUEST_LOG_PATH.open("a", encoding="utf-8") as log_file:
        def request(run, model, effort, data):
            with lock:
                if run in too_slow:
                    return None
            result = run_annotation(
                client, model, effort, template, data,
                questions[data["question_id"]].text, labels,
                sample_path(run, data),
            )
            attempt = {
                "run": run,
                "batch_id": batch_id(data),
                "status": result["status"],
                "error": result.get("error"),
                "elapsed_seconds": result["elapsed_seconds"],
                "cost_usd": completion_cost(result),
            }
            with lock:
                log_file.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                log_file.flush()
                attempts[run].append(attempt)
                too_slow.update(too_slow_runs(attempts))
            return attempt

        pool = ThreadPoolExecutor(CONCURRENT_REQUESTS)
        try:
            futures = [pool.submit(request, *job) for job in jobs]
            completed = 0
            for future in as_completed(futures):
                attempt = future.result()
                if attempt is None:
                    continue
                completed += 1
                seconds = attempt["elapsed_seconds"]
                timing = "" if seconds is None else f" in {seconds:.0f} s"
                failed = attempt["status"] == "failed"
                error = f". {attempt['error']}" if failed else ""
                print(
                    f"[{completed}/{len(jobs)}] {attempt['run']} "
                    f"{attempt['batch_id']}: {attempt['status']}{timing}{error}"
                )
        finally:
            pool.shutdown(cancel_futures=True)

    print(write_report())
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
