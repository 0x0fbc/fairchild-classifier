"""
Compare response quality across Venice.ai models on a varied set of prompts.
"""

from __future__ import annotations

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

from generateResponses import (
    PROMPTS_PATH,
    REQUEST_TIMEOUT_SECONDS,
    ROOT_DIR,
    generate_batch,
    write_json,
)


# Models to compare, in report order. Each uses its default reasoning
# setting: several cannot switch reasoning off, and GLM writes its reasoning
# into the reply when asked to.
MODELS = [
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-sonnet-5",
    "openai-gpt-6-astra",
    "openai-gpt-6-sol",
    "gemini-3-1-pro-preview",
    "grok-4-7",
    "kimi-k3",
    "qwen-3-8-max",
    "deepseek-v4-pro-0813",
    "z-ai-glm-5-3",
    # Baseline: the model used for the full run so far.
    "z-ai-glm-5-3-flash",
]

# Chosen to cover six questions, all three approaches, all four tones, every
# label group, and one to three required labels. Q1_001 (explicit welfare
# disregard, intuitive, energized) is the prompt that produced formulaic
# answers in earlier tests.
SAMPLE_PROMPT_IDS = [
    "Q1_001",
    "Q8_029",
    "Q4_029",
    "Q3_033",
    "Q6_029",
    "Q11_032",
]

CONCURRENT_REQUESTS = 8
SAMPLES_DIR = ROOT_DIR / "data" / "pilots" / "generators"
REPORT_PATH = ROOT_DIR / "results" / "pilots" / "generators.md"
BATCH = 1
WORD = re.compile(r"[\w'’-]+")


def sample(
    client: OpenAI,
    model: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    output_dir = SAMPLES_DIR / model

    try:
        return generate_batch(client, model, record, BATCH, output_dir)
    except openai.APIError as error:
        data = {
            **record,
            "batch": BATCH,
            "model": model,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(
            output_dir / "failed" / f"{record['prompt_id']}_b{BATCH}.json",
            data,
        )
        return data


def load_batch(model: str, prompt_id: str) -> dict[str, Any] | None:
    model_dir = SAMPLES_DIR / model
    name = f"{prompt_id}_b{BATCH}.json"

    for path in (model_dir / name, model_dir / "failed" / name):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    return None


def opening(response: str) -> str:
    return " ".join(WORD.findall(response.lower())[:3])


def summarize(model: str, batches: list[dict[str, Any]]) -> str:
    counts = {status: 0 for status in ("ok", "incompatible", "failed")}
    words: list[int] = []
    distinct: list[float] = []
    seconds: list[float] = []
    reasoning: list[int] = []
    cost = 0.0

    for batch in batches:
        counts[batch["status"]] += 1
        responses = batch.get("responses") or []
        words += [len(WORD.findall(response)) for response in responses]

        if responses:
            openings = {opening(response) for response in responses}
            distinct.append(len(openings) / len(responses))

        if "elapsed_seconds" in batch:
            seconds.append(batch["elapsed_seconds"])

        completion = batch.get("completion") or {}
        usage = completion.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}

        if details.get("reasoning_tokens") is not None:
            reasoning.append(details["reasoning_tokens"])

        cost += (completion.get("cost") or {}).get("usd", 0.0)

    def mean(values: list[float], digits: int = 0) -> str:
        return f"{statistics.mean(values):.{digits}f}" if values else "–"

    median_seconds = f"{statistics.median(seconds):.0f}" if seconds else "–"
    return (
        f"| `{model}` | {counts['ok']} | {counts['incompatible']} "
        f"| {counts['failed']} | {mean(words)} | {mean(distinct, 2)} "
        f"| {median_seconds} | {mean(reasoning)} | {cost:.4f} |"
    )


def describe(batch: dict[str, Any] | None) -> list[str]:
    if batch is None:
        return ["Not generated yet.", ""]

    seconds = batch.get("elapsed_seconds")
    timing = f", {seconds:.0f} s" if seconds is not None else ""

    if batch["status"] == "ok":
        lines = [f"OK{timing}.", ""]
        lines += [
            f"{number}. {response}"
            for number, response in enumerate(batch["responses"], start=1)
        ]
        return [*lines, ""]

    if batch["status"] == "incompatible":
        return [f"Returned INCOMPATIBLE{timing}.", ""]

    lines = [f"Failed{timing}: {batch['error']}", ""]
    completion = batch.get("completion") or {}
    choices = completion.get("choices") or [{}]
    content = (choices[0].get("message") or {}).get("content") or ""

    if content:
        excerpt = content.strip()[:600]
        lines += ["Start of the reply:", "", "```text", excerpt, "```", ""]

    return lines


def write_report(records: list[dict[str, Any]]) -> None:
    lines = [
        "# Model samples",
        "",
        "**Preliminary generator comparison.** Retained answers are in "
        "`data/pilots/generators/` (repository-root-relative). These files "
        "also supply the 300 cross-generator answers used in the main study "
        "and must be retained. This report records the preliminary comparison, "
        "not the main held-out classifier evaluation. See the "
        "[repository guide](../../README.md) for the current manuscript and "
        "reproduction workflow.",
        "",
        "Generated by `src/sampleModels.py`. Each model answered each prompt "
        "once (10 answers), with its default reasoning setting.",
        "",
        "Words: mean words per answer. Openings: distinct first three words "
        "among a batch's answers, as a share of its answers (1.00 means no "
        "two answers start the same way). Seconds: median per request. "
        "Reasoning: mean reasoning tokens per request, where the API reports "
        "them. Cost: total for all saved batches.",
        "",
        "| Model | OK | Incompatible | Failed | Words | Openings | Seconds "
        "| Reasoning | Cost (USD) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for model in MODELS:
        batches = [
            batch
            for record in records
            if (batch := load_batch(model, record["prompt_id"])) is not None
        ]
        lines.append(summarize(model, batches))

    for record in records:
        lines += [
            "",
            f"## {record['prompt_id']}",
            "",
            f"Approach `{record['approach_id']}`, tone `{record['tone_id']}`.",
            "",
            "Required: "
            + ", ".join(f"`{key}`" for key in record["intended_label_keys"]),
            "",
            "Forbidden: "
            + ", ".join(f"`{key}`" for key in record["forbidden_label_keys"]),
            "",
            "<details><summary>Full prompt</summary>",
            "",
            "```text",
            record["prompt"],
            "```",
            "",
            "</details>",
            "",
        ]

        for model in MODELS:
            lines += [f"### `{model}`", ""]
            lines += describe(load_batch(model, record["prompt_id"]))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)

    with PROMPTS_PATH.open(encoding="utf-8") as prompts_file:
        by_id = {
            record["prompt_id"]: record
            for record in map(json.loads, prompts_file)
        }

    missing = [
        prompt_id for prompt_id in SAMPLE_PROMPT_IDS if prompt_id not in by_id
    ]

    if missing:
        raise ValueError(f"Unknown prompt IDs: {missing}")

    records = [by_id[prompt_id] for prompt_id in SAMPLE_PROMPT_IDS]
    jobs = [
        (model, record)
        for record in records
        for model in MODELS
        if not (
            SAMPLES_DIR / model / f"{record['prompt_id']}_b{BATCH}.json"
        ).exists()
    ]
    total = len(records) * len(MODELS)
    print(
        f"{total - len(jobs)} of {total} sample batches already generated "
        f"({len(records)} prompts x {len(MODELS)} models); generating "
        f"{len(jobs)} now, {CONCURRENT_REQUESTS} at a time."
    )

    with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
        futures = {
            pool.submit(sample, client, model, record): (model, record)
            for model, record in jobs
        }

        for number, future in enumerate(as_completed(futures), start=1):
            model, record = futures[future]
            result = future.result()
            status = result["status"]
            detail = (
                result["error"]
                if status == "failed"
                else f"{len(result['responses'])} responses"
            )
            seconds = result.get("elapsed_seconds")
            timing = f" in {seconds:.0f} s" if seconds is not None else ""
            print(
                f"[{number}/{len(jobs)}] {model} {record['prompt_id']}: "
                f"{status}{timing}. {detail}"
            )

    write_report(records)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
