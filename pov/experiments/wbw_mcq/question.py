"""Question model for the word-by-word MCQ experiment.

Accepts the shapes MMLU actually ships in:

* **JSONL, options dict** — ``{"stem": ..., "options": {"A": ...}, "answer": "B"}``
* **JSONL, HuggingFace MMLU** — ``{"question": ..., "choices": [...], "answer": 1}``
* **CSV, original MMLU release** — headerless ``question,A,B,C,D,answer``

`answer` may be a letter (``"B"``) or a 0-based index (``1``); both normalise to
a letter.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LETTERS = ("A", "B", "C", "D")


class QuestionError(ValueError):
    """Raised for malformed question records."""


@dataclass
class Question:
    """One multiple-choice question."""

    id: str
    stem: str
    options: dict
    answer: str
    domain: str = ""
    difficulty: str = ""
    source: str = ""
    word_sequence: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.answer = normalise_answer(self.answer, self.id)
        if not self.stem.strip():
            raise QuestionError(f"{self.id}: stem is empty")
        missing = [letter for letter in LETTERS if not str(self.options.get(letter, "")).strip()]
        if missing:
            raise QuestionError(f"{self.id}: missing option(s) {missing}")
        if self.answer not in self.options:
            raise QuestionError(
                f"{self.id}: answer {self.answer!r} has no matching option"
            )
        if not self.word_sequence:
            self.word_sequence = self.compute_word_sequence()

    # -- derived -----------------------------------------------------------

    def compute_word_sequence(self) -> list:
        """Ordered tokens revealed one at a time.

        Option labels are single tokens (``(A)``) and punctuation stays glued to
        its word, matching the original benchmark's tokenisation.
        """
        tokens: list[str] = list(self.stem.split())
        for letter in LETTERS:
            text = str(self.options.get(letter, "")).strip()
            if text:
                tokens.append(f"({letter})")
                tokens.extend(text.split())
        return tokens

    @property
    def word_count(self) -> int:
        return len(self.word_sequence)

    def format_as_text(self) -> str:
        lines = [self.stem, ""]
        lines.extend(f"({letter}) {self.options.get(letter, '')}" for letter in LETTERS)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stem": self.stem,
            "options": dict(self.options),
            "answer": self.answer,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "source": self.source,
            "word_sequence": list(self.word_sequence),
            "word_count": self.word_count,
        }

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping, index: int = 0,
                  default_id: str | None = None) -> "Question":
        if not isinstance(data, Mapping):
            raise QuestionError(f"record {index}: expected an object, got {type(data).__name__}")

        qid = str(data.get("id") or data.get("qid") or default_id or f"q_{index:04d}")

        stem = data.get("stem")
        if stem is None:
            stem = data.get("question")
        if stem is None:
            raise QuestionError(f"{qid}: no 'stem' or 'question' field")

        options = _extract_options(data, qid)

        if "answer" in data:
            answer = data["answer"]
        elif "correct" in data:
            answer = data["correct"]
        else:
            raise QuestionError(f"{qid}: no 'answer' field")

        return cls(
            id=qid,
            stem=str(stem).strip(),
            options=options,
            answer=answer,
            domain=str(data.get("domain", "") or ""),
            difficulty=str(data.get("difficulty", "") or ""),
            source=str(data.get("source", "") or data.get("subject", "") or ""),
            word_sequence=list(data.get("word_sequence") or []),
        )


def _extract_options(data: Mapping, qid: str) -> dict:
    raw = data.get("options")
    if isinstance(raw, Mapping):
        return {letter: str(raw.get(letter, "")).strip() for letter in LETTERS}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return _options_from_list(raw, qid)

    choices = data.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        return _options_from_list(choices, qid)

    if all(letter in data for letter in LETTERS):
        return {letter: str(data[letter]).strip() for letter in LETTERS}

    raise QuestionError(f"{qid}: no options — expected 'options', 'choices', or A/B/C/D keys")


def _options_from_list(values: Sequence, qid: str) -> dict:
    items = list(values)
    if len(items) != len(LETTERS):
        raise QuestionError(
            f"{qid}: expected {len(LETTERS)} choices, got {len(items)}"
        )
    return {letter: str(value).strip() for letter, value in zip(LETTERS, items)}


def normalise_answer(answer: Any, qid: str = "") -> str:
    """Map a letter, index, or index-like string onto a letter."""
    if isinstance(answer, bool):
        raise QuestionError(f"{qid}: answer must be a letter or index, got a bool")
    if isinstance(answer, int):
        if not 0 <= answer < len(LETTERS):
            raise QuestionError(f"{qid}: answer index {answer} is out of range")
        return LETTERS[answer]

    text = str(answer).strip()
    if not text:
        raise QuestionError(f"{qid}: answer is empty")
    upper = text.upper()
    if upper in LETTERS:
        return upper
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(LETTERS):
            return LETTERS[index]
        # Some dumps are 1-based.
        if 1 <= index <= len(LETTERS):
            return LETTERS[index - 1]
    raise QuestionError(f"{qid}: cannot interpret answer {answer!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_questions(path: str | Path, fmt: str = "auto") -> list[Question]:
    """Load questions from a .jsonl or MMLU .csv file."""
    path = Path(path)
    if not path.exists():
        raise QuestionError(f"questions file not found: {path}")

    if fmt == "auto":
        fmt = "mmlu_csv" if path.suffix.lower() == ".csv" else "jsonl"

    if fmt == "jsonl":
        questions = _load_jsonl(path)
    elif fmt == "mmlu_csv":
        questions = _load_mmlu_csv(path)
    else:
        raise QuestionError(f"unknown questions format {fmt!r} (use auto, jsonl, or mmlu_csv)")

    if not questions:
        raise QuestionError(f"{path}: no questions found")

    duplicates = _duplicates(q.id for q in questions)
    if duplicates:
        raise QuestionError(f"{path}: duplicate question id(s) {sorted(duplicates)[:5]}")
    return questions


def _load_jsonl(path: Path) -> list[Question]:
    records: list[tuple[int, Mapping]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise QuestionError(f"{path}:{line_no}: invalid JSON — {exc}") from exc

    # Ids are assigned in two passes: explicit ids are reserved first, so an
    # auto-generated id can never collide with one the file already uses.
    taken = {
        str(record.get("id") or record.get("qid"))
        for _, record in records
        if isinstance(record, Mapping) and (record.get("id") or record.get("qid"))
    }

    questions: list[Question] = []
    counter = 0
    for index, (line_no, record) in enumerate(records):
        default_id = None
        if isinstance(record, Mapping) and not (record.get("id") or record.get("qid")):
            while True:
                default_id = f"q_{counter:04d}"
                counter += 1
                if default_id not in taken:
                    taken.add(default_id)
                    break
        try:
            questions.append(Question.from_dict(record, index=index, default_id=default_id))
        except QuestionError as exc:
            raise QuestionError(f"{path}:{line_no}: {exc}") from exc
    return questions


def _load_mmlu_csv(path: Path) -> list[Question]:
    questions: list[Question] = []
    subject = path.stem.replace("_test", "").replace("_val", "").replace("_", " ")
    with open(path, encoding="utf-8", newline="") as f:
        for row_no, row in enumerate(csv.reader(f), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 6:
                raise QuestionError(
                    f"{path}:{row_no}: expected 6 columns "
                    f"(question,A,B,C,D,answer), got {len(row)}"
                )
            # A header row would fail answer normalisation; skip it explicitly.
            if row_no == 1 and row[-1].strip().lower() in ("answer", "label", "target"):
                continue
            index = len(questions)
            try:
                questions.append(
                    Question(
                        id=f"q_{index:04d}",
                        stem=row[0].strip(),
                        options={letter: row[i + 1].strip() for i, letter in enumerate(LETTERS)},
                        answer=row[5].strip(),
                        source=subject,
                    )
                )
            except QuestionError as exc:
                raise QuestionError(f"{path}:{row_no}: {exc}") from exc
    return questions


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes
