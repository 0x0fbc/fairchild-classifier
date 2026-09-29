"""
Generate synthetic responses for the prompts in generation_prompts.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI


# Adjust these to control the generation run.
# Models to generate with, each mapped to the reasoning effort sent with its
# requests (None leaves the model's default). Medium is Venice's default for
# claude-opus-5-5; it is set explicitly so it is visible and fixed.
# claude-sonnet-5 keeps the default it was sampled with (medium).
GENERATION_MODELS: dict[str, str | None] = {
    "claude-opus-5-5": "medium",
    "claude-sonnet-5": None,
}
# Independent requests per prompt, each returning the prompt's response_count
# responses. Smaller requests are meant to reduce templated answers.
BATCHES_PER_PROMPT = 2
# Requests to send per run. None sends every remaining one.
REQUESTS_PER_RUN: int | None = None
# Requests in flight at once.
CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT_SECONDS = 1200

# venice.ai-specific request options; remove them when calling another provider.
# - include_venice_system_prompt: Venice prepends its own system prompt
#   unless this is disabled.
EXTRA_BODY = {
    "venice_parameters": {"include_venice_system_prompt": False},
}

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT_DIR / "data" / "main_corpus" / "generation_prompts.jsonl"
RAW_DIR = ROOT_DIR / "data" / "main_corpus" / "responses"

INCOMPATIBLE = "INCOMPATIBLE"
CODE_FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*)\n[ \t]*```", re.DOTALL)


class ParseError(ValueError):
    """Raised when model output is neither the JSON array nor INCOMPATIBLE."""


def parse_responses(
    content: str | None,
    response_count: int,
) -> list[str] | None:
    text = (content or "").strip()

    if fence := CODE_FENCE.fullmatch(text):
        text = fence.group(1).strip()

    if text == INCOMPATIBLE:
        return None

    try:
        responses = json.loads(text)
    except json.JSONDecodeError as error:
        raise ParseError(f"Output is not valid JSON: {error}") from None

    if not isinstance(responses, list) or not all(
        isinstance(response, str) and response.strip()
        for response in responses
    ):
        raise ParseError("Output is not a JSON array of non-empty strings.")

    if len(responses) != response_count:
        raise ParseError(
            f"Expected {response_count} responses, got {len(responses)}."
        )

    return responses


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def generate_batch(
    client: OpenAI,
    model: str,
    record: dict[str, Any],
    batch: int,
    output_dir: Path,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    batch_id = f"{record['prompt_id']}_b{batch}"
    failed_path = output_dir / "failed" / f"{batch_id}.json"
    started = time.monotonic()
    options: dict[str, Any] = {}

    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": record["prompt"]}],
        extra_body=EXTRA_BODY,
        **options,
    )
    data: dict[str, Any] = {
        **record,
        "batch": batch,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }

    try:
        responses = parse_responses(
            completion.choices[0].message.content,
            record["response_count"],
        )
    except ParseError as error:
        data |= {
            "status": "failed",
            "error": str(error),
            "completion": completion.to_dict(mode="json"),
        }
        write_json(failed_path, data)
        return data

    data |= {
        "status": "ok" if responses is not None else "incompatible",
        "responses": responses or [],
        "completion": completion.to_dict(mode="json"),
    }
    write_json(output_dir / f"{batch_id}.json", data)
    failed_path.unlink(missing_ok=True)
    return data


def run_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str | None,
    record: dict[str, Any],
    batch: int,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        return generate_batch(
            client,
            model,
            record,
            batch,
            output_dir,
            reasoning_effort=reasoning_effort,
        )
    except openai.APIError as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": None,
        }


def generate_pending(
    client: OpenAI, records: list[dict[str, Any]],
    raw_dir: Path, batches_per_prompt: int,
) -> None:

    finished = {
        (model, path.stem)
        for model in GENERATION_MODELS
        for path in (raw_dir / model).glob("*.json")
    }
    jobs = [
        (model, effort, record, batch, f"{record['prompt_id']}_b{batch}")
        for model, effort in GENERATION_MODELS.items()
        for record in records
        for batch in range(1, batches_per_prompt + 1)
    ]
    pending = [job for job in jobs if (job[0], job[4]) not in finished]
    run = pending if REQUESTS_PER_RUN is None else pending[:REQUESTS_PER_RUN]

    print(
        f"{len(jobs) - len(pending)} of {len(jobs)} batches "
        f"({len(records)} prompts x {batches_per_prompt} batches x "
        f"{len(GENERATION_MODELS)} models) already generated; generating "
        f"{len(run)} now, {CONCURRENT_REQUESTS} at a time."
    )

    generated = 0
    failed = 0
    pool = ThreadPoolExecutor(CONCURRENT_REQUESTS)
    futures = {
        pool.submit(
            run_batch, client, model, effort, record, batch, raw_dir / model
        ): f"{model}/{batch_id}"
        for model, effort, record, batch, batch_id in run
    }

    try:
        for number, future in enumerate(as_completed(futures), start=1):
            progress = f"[{number}/{len(run)}] {futures[future]}"
            result = future.result()
            elapsed = result["elapsed_seconds"]

            if result["status"] == "failed":
                failed += 1
                after = "" if elapsed is None else f" after {elapsed:.0f} s"
                print(f"{progress}: failed{after}. {result['error']}")
                continue

            generated += 1
            print(
                f"{progress}: {result['status']}, "
                f"{len(result['responses'])} responses in {elapsed:.0f} s"
            )
    finally:
        pool.shutdown(cancel_futures=True)

    print(
        f"Generated {generated} and failed {failed} of {len(run)} batches; "
        f"{len(pending) - generated} remaining. Output: {raw_dir}"
    )


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)

    with PROMPTS_PATH.open(encoding="utf-8") as prompts_file:
        records = [json.loads(line) for line in prompts_file]
    generate_pending(client, records, RAW_DIR, BATCHES_PER_PROMPT)


if __name__ == "__main__":
    main()
