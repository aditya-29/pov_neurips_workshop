#!/usr/bin/env python3
"""Fetch real MMLU questions for the word-by-word MCQ experiment.

    python scripts/fetch_mmlu.py                       # 200 questions, all subjects
    python scripts/fetch_mmlu.py --limit 500
    python scripts/fetch_mmlu.py --subjects anatomy astronomy world_religions
    python scripts/fetch_mmlu.py --split validation --out data/questions_val.jsonl

Downloads MMLU from the HuggingFace Hub and writes `data/questions.jsonl` in the
shape `pov`'s loader expects. Sampling is stratified across subjects and seeded,
so the same flags always produce the same file.

Requires `datasets`:  pip install datasets
(Or use --from-csv to convert a local copy of the original MMLU release instead,
which needs no extra dependency and no network.)

MMLU: Hendrycks et al., "Measuring Massive Multitask Language Understanding",
ICLR 2021. Released under the MIT licence.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "questions.jsonl"
DEFAULT_DATASET = "cais/mmlu"
LETTERS = ("A", "B", "C", "D")

# MMLU subjects grouped the way the original design doc split domains.
_STEM_HINTS = (
    "math", "physics", "chemistry", "biology", "computer", "engineering",
    "statistics", "astronomy", "econometrics", "electrical",
)
_MEDICAL_HINTS = ("medicine", "medical", "anatomy", "clinical", "virology", "nutrition")
_HUMANITIES_HINTS = (
    "history", "philosophy", "law", "religions", "moral", "logic", "prehistory",
)


def infer_domain(subject: str) -> str:
    text = subject.lower()
    if any(hint in text for hint in _MEDICAL_HINTS):
        return "medical"
    if any(hint in text for hint in _STEM_HINTS):
        return "stem"
    if any(hint in text for hint in _HUMANITIES_HINTS):
        return "humanities"
    return "commonsense"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch MMLU questions into pov's questions.jsonl format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output .jsonl path")
    parser.add_argument("--limit", type=int, default=200,
                        help="total questions to write (0 = keep everything)")
    parser.add_argument("--split", default="test",
                        choices=("test", "validation", "dev", "auxiliary_train"),
                        help="MMLU split to draw from")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="restrict to these MMLU subjects (default: all)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="sampling seed")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="HuggingFace dataset id")
    parser.add_argument("--max-words", type=int, default=120,
                        help="skip questions longer than this many words "
                             "(a 300-word stem makes a very long video); 0 = no cap")
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="convert local MMLU .csv files in this directory "
                             "instead of downloading (no dependencies needed)")
    return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────────────


def load_from_hub(dataset: str, split: str, subjects: list[str] | None) -> list[dict]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        sys.exit(
            "error: the 'datasets' package is required to download MMLU.\n"
            "       pip install datasets\n"
            "       (or use --from-csv with a local copy of the MMLU release)"
        )

    if subjects:
        configs = subjects
    else:
        print(f"listing subjects in {dataset}…", flush=True)
        configs = [c for c in get_dataset_config_names(dataset) if c != "all"]

    records: list[dict] = []
    for index, subject in enumerate(configs, start=1):
        print(f"  [{index}/{len(configs)}] {subject}", flush=True)
        try:
            data = load_dataset(dataset, subject, split=split)
        except Exception as exc:
            print(f"      skipped ({exc})", file=sys.stderr)
            continue
        for row in data:
            records.append({
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": row["answer"],
                "subject": row.get("subject") or subject,
            })
    return records


def load_from_csv(directory: Path, subjects: list[str] | None) -> list[dict]:
    """Read the original MMLU release: headerless question,A,B,C,D,answer CSVs."""
    if not directory.is_dir():
        sys.exit(f"error: --from-csv path is not a directory: {directory}")

    paths = sorted(directory.glob("*.csv"))
    if not paths:
        sys.exit(f"error: no .csv files found in {directory}")

    records: list[dict] = []
    for path in paths:
        subject = path.stem.replace("_test", "").replace("_val", "").replace("_dev", "")
        if subjects and subject not in subjects:
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6 or not row[0].strip():
                    continue
                if row[5].strip().lower() in ("answer", "label", "target"):
                    continue  # header row
                records.append({
                    "question": row[0].strip(),
                    "choices": [cell.strip() for cell in row[1:5]],
                    "answer": row[5].strip(),
                    "subject": subject,
                })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Shaping
# ──────────────────────────────────────────────────────────────────────────────


def to_question(record: dict, index: int) -> dict | None:
    """Convert a raw record into pov's question schema, or None if unusable."""
    stem = str(record.get("question", "")).strip()
    choices = [str(choice).strip() for choice in record.get("choices", [])]
    if not stem or len(choices) != 4 or not all(choices):
        return None

    answer = record.get("answer")
    if isinstance(answer, int):
        if not 0 <= answer < 4:
            return None
        letter = LETTERS[answer]
    else:
        text = str(answer).strip().upper()
        if text in LETTERS:
            letter = text
        elif text.isdigit() and 0 <= int(text) < 4:
            letter = LETTERS[int(text)]
        else:
            return None

    subject = str(record.get("subject", "")).strip()
    return {
        "id": f"q_{index:05d}",
        "domain": infer_domain(subject),
        "difficulty": "",
        "source": subject or "MMLU",
        "stem": stem,
        "options": dict(zip(LETTERS, choices)),
        "answer": letter,
    }


def word_count(question: dict) -> int:
    total = len(question["stem"].split())
    for option in question["options"].values():
        total += len(option.split()) + 1  # +1 for the "(A)" label token
    return total


def stratified_sample(questions: list[dict], limit: int, seed: int) -> list[dict]:
    """Sample evenly across subjects, spreading the remainder."""
    if limit <= 0 or limit >= len(questions):
        return questions

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        by_subject[question["source"]].append(question)

    rng = random.Random(seed)
    for pool in by_subject.values():
        pool.sort(key=lambda q: q["id"])
        rng.shuffle(pool)

    chosen: list[dict] = []
    subjects = sorted(by_subject)
    # Round-robin across subjects until the quota is met, so no subject
    # dominates and small subjects are not squeezed out entirely.
    depth = 0
    while len(chosen) < limit:
        added = False
        for subject in subjects:
            pool = by_subject[subject]
            if depth < len(pool):
                chosen.append(pool[depth])
                added = True
                if len(chosen) == limit:
                    break
        if not added:
            break
        depth += 1
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.from_csv:
        raw = load_from_csv(args.from_csv, args.subjects)
        origin = str(args.from_csv)
    else:
        raw = load_from_hub(args.dataset, args.split, args.subjects)
        origin = f"{args.dataset}:{args.split}"

    if not raw:
        sys.exit("error: no questions were loaded")

    questions: list[dict] = []
    skipped_malformed = 0
    for record in raw:
        question = to_question(record, len(questions))
        if question is None:
            skipped_malformed += 1
            continue
        questions.append(question)

    skipped_long = 0
    if args.max_words > 0:
        kept = [q for q in questions if word_count(q) <= args.max_words]
        skipped_long = len(questions) - len(kept)
        questions = kept

    if not questions:
        sys.exit("error: every question was filtered out — try raising --max-words")

    questions = stratified_sample(questions, args.limit, args.seed)
    # Renumber so ids are contiguous in the written file.
    for index, question in enumerate(questions):
        question["id"] = f"q_{index:05d}"
    questions.sort(key=lambda q: q["id"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")

    subjects = sorted({q["source"] for q in questions})
    lengths = sorted(word_count(q) for q in questions)
    print()
    print(f"wrote {len(questions)} questions -> {args.out}")
    print(f"  source        : {origin}")
    print(f"  subjects      : {len(subjects)}")
    print(f"  words/question: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}")
    if skipped_malformed:
        print(f"  skipped (malformed): {skipped_malformed}")
    if skipped_long:
        print(f"  skipped (> {args.max_words} words): {skipped_long}")
    print()
    print("Generate with:")
    print(f"  pov generate -c configs/wbw_mcq.yaml --set params.questions_path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
