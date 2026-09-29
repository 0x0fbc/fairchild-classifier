"""
Build the classifier dataset from the generated response batches.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from generatePrompts import REQUIRES, Label, load_prompt_data
from generateResponses import ROOT_DIR


# Models whose data/main_corpus/responses/<model>/ batches are training data.
TRAINING_GENERATORS = ("claude-opus-5-5", "claude-sonnet-5")
RAW_DIR = ROOT_DIR / "data" / "main_corpus" / "responses"
# The annotator src/sampleAnnotators.py selected
ANNOTATOR_MODEL = "claude-opus-5-5"
ANNOTATIONS_ROOT = ROOT_DIR / "data" / "main_corpus" / "annotations"
# A second, independent rater for the test split, standing in for a second
# human rater. It gets the same annotation prompt as ANNOTATOR_MODEL
SECOND_RATER_MODEL = "gemini-3-1-pro-preview"
SECOND_RATER_SPLIT = "test"
SAMPLES_DIR = ROOT_DIR / "data" / "pilots" / "generators"
DATASET_PATH = ROOT_DIR / "data" / "main_corpus" / "classifier_data.jsonl"
# Whole dilemmas are held out, so scores measure generalisation to unseen
# scenarios.
CALIBRATION_QUESTIONS = {"Q5", "Q13"}
TEST_QUESTIONS = {"Q3", "Q6", "Q11", "Q14"}
SPLITS = ("train", "calibration", "test", "cross_generator")


def load_labels() -> list[Label]:
    return list(load_prompt_data()[0].values())


def load_rows(splits: set[str]) -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found; run src/buildClassifierData.py first."
        )

    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return [
            row
            for row in map(json.loads, dataset_file)
            if row["split"] in splits
        ]


def question_split(question_id: str) -> str:
    if question_id in TEST_QUESTIONS:
        return "test"

    if question_id in CALIBRATION_QUESTIONS:
        return "calibration"

    return "train"


def source_batches() -> list[tuple[dict[str, Any], str]]:
    batches: list[tuple[dict[str, Any], str]] = []

    for model in TRAINING_GENERATORS:
        for path in sorted((RAW_DIR / model).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))

            if data["status"] == "ok":
                batches.append((data, question_split(data["question_id"])))

    for model_dir in sorted(SAMPLES_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name in TRAINING_GENERATORS:
            continue

        for path in sorted(model_dir.glob("*_b1.json")):
            data = json.loads(path.read_text(encoding="utf-8"))

            if data["status"] == "ok" and data["question_id"] in TEST_QUESTIONS:
                batches.append((data, "cross_generator"))

    return batches


def annotation_path(
    data: dict[str, Any], annotator: str = ANNOTATOR_MODEL,
    *, root: Path = ANNOTATIONS_ROOT,
) -> Path:
    return (
        root
        / annotator
        / data["model"]
        / f"{data['prompt_id']}_b{data['batch']}.json"
    )


def annotated_label_keys(annotation: Iterable[str]) -> list[str]:
    keys = set(annotation)

    prerequisites = {
        prerequisite for key in keys for prerequisite in REQUIRES.get(key, set())
    }
    return sorted(keys | prerequisites)


def batch_rows(
    data: dict[str, Any],
    split: str,
    annotations: list[dict[str, str]],
    second_rater_annotations: list[dict[str, str]] | None,
) -> list[dict[str, Any]]:
    positives = sorted(data["intended_label_keys"])
    negatives = sorted(data["forbidden_label_keys"])
    second_rater_keys = (
        [None] * len(data["responses"])
        if second_rater_annotations is None
        else [
            annotated_label_keys(annotation)
            for annotation in second_rater_annotations
        ]
    )

    return [
        {
            "id": (
                f"{data['model']}/{data['prompt_id']}"
                f"_b{data['batch']}_r{index:02d}"
            ),
            "text": text,
            "generator_model": data["model"],
            "prompt_id": data["prompt_id"],
            "question_id": data["question_id"],
            "batch": data["batch"],
            "response_index": index,
            "approach_id": data["approach_id"],
            "tone_id": data["tone_id"],
            "positive_label_keys": positives,
            "negative_label_keys": negatives,
            "annotated_label_keys": annotated_label_keys(annotation),
            "second_rater_label_keys": second_rater,
            "split": split,
        }
        for index, (text, annotation, second_rater) in enumerate(
            zip(
                data["responses"],
                annotations,
                second_rater_keys,
                strict=True,
            ),
            start=1,
        )
    ]


def validate_rows(rows: list[dict[str, Any]], label_keys: set[str]) -> None:
    seen_ids: set[str] = set()

    for row in rows:
        row_id = row["id"]
        positives = set(row["positive_label_keys"])
        negatives = set(row["negative_label_keys"])
        annotated = set(row["annotated_label_keys"])
        second_rater = set(row["second_rater_label_keys"] or [])

        if unknown := (
            positives | negatives | annotated | second_rater
        ) - label_keys:
            raise ValueError(
                f"{row_id}: label keys not in codebook.csv: {sorted(unknown)}"
            )

        if overlap := positives & negatives:
            raise ValueError(
                f"{row_id}: label keys both positive and negative: "
                f"{sorted(overlap)}"
            )

        for key in sorted(positives):
            if missing := REQUIRES.get(key, set()) - positives:
                raise ValueError(
                    f"{row_id}: positive {key} lacks its prerequisite "
                    f"{sorted(missing)}"
                )

        if not positives:
            raise ValueError(f"{row_id}: no positive label keys")

        if not row["text"].strip():
            raise ValueError(f"{row_id}: empty text")

        if row_id in seen_ids:
            raise ValueError(f"{row_id}: duplicate id")

        seen_ids.add(row_id)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary_path.replace(path)


def main() -> None:
    batches = source_batches()
    missing = sum(not annotation_path(data).exists() for data, _ in batches)

    if missing:
        raise FileNotFoundError(
            f"{missing} of {len(batches)} batches have no annotation in "
            f"{ANNOTATIONS_ROOT / ANNOTATOR_MODEL}; run "
            f"src/annotateResponses.py first."
        )

    second_rater_batches = [
        data for data, split in batches if split == SECOND_RATER_SPLIT
    ]
    missing = sum(
        not annotation_path(data, SECOND_RATER_MODEL).exists()
        for data in second_rater_batches
    )

    if missing:
        raise FileNotFoundError(
            f"{missing} of {len(second_rater_batches)} {SECOND_RATER_SPLIT} "
            f"batches have no annotation by {SECOND_RATER_MODEL} in "
            f"{ANNOTATIONS_ROOT / SECOND_RATER_MODEL}; run "
            f"src/annotateResponses.py first."
        )

    rows: list[dict[str, Any]] = []

    for data, split in batches:
        annotations = json.loads(
            annotation_path(data).read_text(encoding="utf-8")
        )["annotations"]
        second_rater_annotations = None

        if split == SECOND_RATER_SPLIT:
            second_rater_annotations = json.loads(
                annotation_path(data, SECOND_RATER_MODEL).read_text(
                    encoding="utf-8"
                )
            )["annotations"]

        rows.extend(
            batch_rows(data, split, annotations, second_rater_annotations)
        )

    validate_rows(rows, {label.machine_key for label in load_labels()})
    write_jsonl(DATASET_PATH, rows)

    split_counts = Counter(row["split"] for row in rows)
    generator_counts = Counter(row["generator_model"] for row in rows)
    second_rated = sum(
        row["second_rater_label_keys"] is not None for row in rows
    )
    designed = 0
    disagreements = 0

    for row in rows:
        positives = set(row["positive_label_keys"])
        negatives = set(row["negative_label_keys"])
        annotated = set(row["annotated_label_keys"])
        designed += len(positives) + len(negatives)
        disagreements += len(positives - annotated) + len(negatives & annotated)

    print(f"Wrote {len(rows)} rows to {DATASET_PATH}")

    for split in SPLITS:
        print(f"  {split}: {split_counts[split]} responses")

    for model, count in sorted(generator_counts.items()):
        print(f"  {model}: {count} responses")

    print(
        f"  {disagreements} of {designed} designed pairs disagree with the "
        f"annotation"
    )
    print(
        f"  {second_rated} {SECOND_RATER_SPLIT} responses carry "
        f"{SECOND_RATER_MODEL}'s annotation"
    )


if __name__ == "__main__":
    main()
