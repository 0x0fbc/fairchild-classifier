from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template
from openai import OpenAI

from buildClassifierData import (
    ANNOTATOR_MODEL,
    ANNOTATIONS_ROOT,
    SECOND_RATER_MODEL,
    SECOND_RATER_SPLIT,
    annotated_label_keys,
    annotation_path,
    load_labels,
    source_batches,
)
from generatePrompts import Label, Question, load_prompt_data
from generateResponses import (
    CODE_FENCE,
    EXTRA_BODY,
    REQUEST_TIMEOUT_SECONDS,
    ROOT_DIR,
    ParseError,
    write_json,
)


# Adjust these to control the annotation run.
# Sent as reasoning_effort
ANNOTATOR_REASONING_EFFORT: str | None = "low"
# The second rater runs at its default effort
SECOND_RATER_REASONING_EFFORT: str | None = None
# Requests to send per run. None sends every remaining one.
REQUESTS_PER_RUN: int | None = None
CONCURRENT_REQUESTS = 10
SEED = 1
TEMPLATE_NAME = "annotation.jinja2"


def load_template() -> Template:
    environment = Environment(
        loader=FileSystemLoader(ROOT_DIR / "materials" / "prompts"),
        autoescape=False,
    )
    return environment.get_template(TEMPLATE_NAME)


def parse_annotations(
    content: str | None,
    response_count: int,
    label_keys: set[str],
) -> list[dict[str, str]]:
    text = (content or "").strip()

    if fence := CODE_FENCE.fullmatch(text):
        text = fence.group(1).strip()

    try:
        annotations = json.loads(text)
    except json.JSONDecodeError as error:
        raise ParseError(f"Output is not valid JSON: {error}") from None

    if (
        not isinstance(annotations, list)
        or len(annotations) != response_count
        or not all(isinstance(annotation, dict) for annotation in annotations)
    ):
        raise ParseError(f"Expected a JSON array of {response_count} objects.")

    keys = {key for annotation in annotations for key in annotation}

    if unknown := keys - label_keys:
        raise ParseError(f"Unknown label keys: {sorted(unknown)}")

    if not all(
        isinstance(evidence, str)
        for annotation in annotations
        for evidence in annotation.values()
    ):
        raise ParseError("Evidence must be strings.")

    return annotations


def annotate_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str | None,
    template: Template,
    data: dict[str, Any],
    question_text: str,
    labels: list[Label],
    output_path: Path,
) -> dict[str, Any]:
    failed_path = output_path.parent / "failed" / output_path.name
    prompt = template.render(
        questionText=question_text,
        labels=labels,
        responses=data["responses"],
    )
    started = time.monotonic()
    options: dict[str, Any] = {}

    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body=EXTRA_BODY,
        **options,
    )
    result: dict[str, Any] = {
        "generator_model": data["model"],
        "prompt_id": data["prompt_id"],
        "question_id": data["question_id"],
        "batch": data["batch"],
        "intended_label_keys": data["intended_label_keys"],
        "forbidden_label_keys": data["forbidden_label_keys"],
        "annotator_model": model,
        "reasoning_effort": reasoning_effort,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }

    try:
        annotations = parse_annotations(
            completion.choices[0].message.content,
            len(data["responses"]),
            {label.machine_key for label in labels},
        )
    except ParseError as error:
        result |= {
            "status": "failed",
            "error": str(error),
            "completion": completion.to_dict(mode="json"),
        }
        write_json(failed_path, result)
        return result

    result |= {
        "status": "ok",
        "annotations": annotations,
        "completion": completion.to_dict(mode="json"),
    }
    write_json(output_path, result)
    failed_path.unlink(missing_ok=True)
    return result


def run_annotation(
    client: OpenAI,
    model: str,
    reasoning_effort: str | None,
    template: Template,
    data: dict[str, Any],
    question_text: str,
    labels: list[Label],
    output_path: Path,
) -> dict[str, Any]:
    try:
        return annotate_batch(
            client,
            model,
            reasoning_effort,
            template,
            data,
            question_text,
            labels,
            output_path,
        )
    except openai.APIError as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": None,
        }


def completion_cost(result: dict[str, Any]) -> float:
    completion = result.get("completion") or {}
    return (completion.get("cost") or {}).get("usd", 0.0)


def design_agreement(
    batches: list[tuple[dict[str, Any], list[dict[str, str]]]],
) -> tuple[float, int, float, int]:
    intended = intended_present = forbidden = forbidden_absent = 0

    for data, annotations in batches:
        intended_keys = set(data["intended_label_keys"])
        forbidden_keys = set(data["forbidden_label_keys"])

        for annotation in annotations:
            present = set(annotated_label_keys(annotation))
            intended += len(intended_keys)
            intended_present += len(intended_keys & present)
            forbidden += len(forbidden_keys)
            forbidden_absent += len(forbidden_keys - present)

    return (
        intended_present / intended if intended else math.nan,
        intended,
        forbidden_absent / forbidden if forbidden else math.nan,
        forbidden,
    )


def print_summary(
    annotator: str, batches: list[dict[str, Any]],
    root: Path = ANNOTATIONS_ROOT,
) -> None:
    annotated: list[tuple[dict[str, Any], list[dict[str, str]]]] = []
    cost = 0.0
    present = 0
    responses = 0

    for data in batches:
        path = annotation_path(data, annotator, root=root)

        if not path.exists():
            continue

        result = json.loads(path.read_text(encoding="utf-8"))
        annotated.append((data, result["annotations"]))
        cost += completion_cost(result)
        present += sum(
            len(annotated_label_keys(annotation))
            for annotation in result["annotations"]
        )
        responses += len(result["annotations"])

    done = len(annotated)
    total = len(batches)

    if not done:
        print(
            f"{annotator}: annotated 0 of {total} batches; {total} remaining."
        )
        return

    intended_share, intended, forbidden_share, forbidden = design_agreement(
        annotated
    )
    print(
        f"{annotator}: annotated {done} of {total} batches; "
        f"{total - done} remaining. Cost ${cost:.2f} "
        f"(${cost / done:.4f} per request)."
    )
    print(
        f"{annotator}: design agreement {intended_share:.3f} of {intended} "
        f"intended pairs marked present; {forbidden_share:.3f} of "
        f"{forbidden} forbidden pairs marked absent. "
        f"{present / responses:.1f} labels marked present per response."
    )


def annotate_pending(
    client: OpenAI, template: Template, labels: list[Label],
    questions: dict[str, Question],
    annotators: list[tuple[str, str | None, list[dict[str, Any]]]],
    root: Path,
) -> None:
    total = sum(
        len(annotator_batches) for _, _, annotator_batches in annotators
    )
    pending = [
        (model, effort, data)
        for model, effort, annotator_batches in annotators
        for data in annotator_batches
        if not annotation_path(data, model, root=root).exists()
    ]
    # Shuffled so that a partial run covers every dilemma and generator.
    random.Random(SEED).shuffle(pending)
    run = pending if REQUESTS_PER_RUN is None else pending[:REQUESTS_PER_RUN]

    print(
        f"{total - len(pending)} of {total} annotations already saved "
        f"({', '.join(f'{len(b)} batches by {m}' for m, _, b in annotators)}); "
        f"annotating {len(run)} now, "
        f"{CONCURRENT_REQUESTS} at a time."
    )

    pool = ThreadPoolExecutor(CONCURRENT_REQUESTS)
    futures = {
        pool.submit(
            run_annotation,
            client,
            model,
            effort,
            template,
            data,
            questions[data["question_id"]].text,
            labels,
            annotation_path(data, model, root=root),
        ): f"{model} {data['model']}/{data['prompt_id']}_b{data['batch']}"
        for model, effort, data in run
    }

    try:
        for number, future in enumerate(as_completed(futures), start=1):
            progress = f"[{number}/{len(run)}] {futures[future]}"
            result = future.result()
            elapsed = result["elapsed_seconds"]

            if result["status"] == "failed":
                after = "" if elapsed is None else f" after {elapsed:.0f} s"
                print(f"{progress}: failed{after}. {result['error']}")
                continue

            print(f"{progress}: ok in {elapsed:.0f} s")
    finally:
        pool.shutdown(cancel_futures=True)

    for model, _, annotator_batches in annotators:
        print_summary(model, annotator_batches, root)


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)
    template = load_template()
    labels = load_labels()
    questions = load_prompt_data()[1]

    batches = source_batches()
    second_rater_batches = [
        data for data, split in batches if split == SECOND_RATER_SPLIT
    ]
    annotators = [
        (
            ANNOTATOR_MODEL,
            ANNOTATOR_REASONING_EFFORT,
            [data for data, _ in batches],
        ),
        (
            SECOND_RATER_MODEL,
            SECOND_RATER_REASONING_EFFORT,
            second_rater_batches,
        ),
    ]
    annotate_pending(client, template, labels, questions, annotators, root=ANNOTATIONS_ROOT)


if __name__ == "__main__":
    main()
