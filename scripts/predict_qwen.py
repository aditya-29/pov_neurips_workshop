#!/usr/bin/env python
"""Fill `model_output` on a pov manifest using a local Qwen3.5 VL model.

pov itself never calls a model — this is the step-5 "Predict" bridge between
`pov generate` and `pov eval`. Reads a run's manifest.jsonl, sends each media
file plus the condition's prompt to the model, and writes preds.jsonl.

    python scripts/predict_qwen.py --run-dir data/chess/<run_id> --limit 4

Every frame-sampling parameter lands in the output as a `sample_*` field, so a
preds file always says how its videos were sampled. That matters here: this
benchmark varies clip *duration*, and a fixed frame budget is itself an
experimental choice rather than an implementation detail.

Ground truth is never sent to the model. `ground_truth`, `source_name`, and
`source_path` are read only to be copied through to the output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Fields that contain or point at the answer. Never put these in a prompt.
LEAK_FIELDS = ("ground_truth", "ground_truth_path", "source_name", "source_path")

# Per-experiment frame sampling. One global fps cannot serve all three, because
# the content moves at wildly different rates:
#
#   chess    static holds of 36 frames (1.2 s) per position, so ~1 fps catches
#            every position in short clips; long clips hit the cap by design.
#   asl      continuous human signing. Handshape and movement are the signal, so
#            a low rate destroys the task outright — at 1 fps a 7 s clip yields
#            4 frames, which no interpreter could read.
#   wbw_mcq  one word every 6 frames (0.2 s) at the `fast` speed, so anything
#            under ~5 fps silently drops most of the question's words.
#
# These are floors for the task to be *possible*; max_frames then bounds cost.
SAMPLING = {
    "chess":   {"fps": 1.0,  "max_frames": 96,  "min_frames": 4},
    "asl":     {"fps": 6.0,  "max_frames": 128, "min_frames": 8},
    "wbw_mcq": {"fps": 10.0, "max_frames": 128, "min_frames": 4},
}
DEFAULT_SAMPLING = {"fps": 2.0, "max_frames": 96, "min_frames": 4}

# Output budget, per experiment. Chess must *emit* one line per half-move, so a
# fixed 512 caps `moves_predicted` at ~43 and makes every clip over ~2 min score
# the same — conflating "could not see the moves" with "was cut off mid-sentence".
# A 10 min clip holds ~495 half-moves at ~12 tokens per line, hence ~6k.
# wbw_mcq is MMLU: the model reasons through the question before answering, and
# a hard stem (e.g. the minimal polynomial of sqrt 6) runs past 256 tokens, so it
# is cut off before emitting "ANSWER: X". That scores as `answered=0` -- a refusal
# the model never made. Budget for the reasoning, not just the letter.
MAX_NEW_TOKENS = {"chess": 8192, "asl": 512, "wbw_mcq": 1024}
DEFAULT_MAX_NEW_TOKENS = 512


def repetition_penalty_for(experiment: str, args) -> float:
    """The flag is the only source: repetition handling stays opt-in per run."""
    return args.repetition_penalty


def max_new_tokens_for(experiment: str, args) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    return MAX_NEW_TOKENS.get(experiment, DEFAULT_MAX_NEW_TOKENS)


def sampling_for(experiment: str, args) -> dict:
    """Per-experiment defaults, with any explicit CLI flag winning."""
    s = dict(SAMPLING.get(experiment, DEFAULT_SAMPLING))
    if args.fps is not None:
        s["fps"] = args.fps
    if args.max_frames is not None:
        s["max_frames"] = args.max_frames
    if args.min_frames is not None:
        s["min_frames"] = args.min_frames
    return s


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="run directory containing manifest.jsonl")
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest path (default: <run-dir>/manifest.jsonl)")
    p.add_argument("--output", type=Path, default=None,
                   help="output path (default: <run-dir>/preds.jsonl)")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N rows (after --stratify)")
    p.add_argument("--stratify", type=int, default=None, metavar="N",
                   help="take N rows per condition — a pilot that covers every "
                        "condition instead of the first N rows of one")
    p.add_argument("--conditions", default=None,
                   help="comma-separated conditions to keep")
    p.add_argument("--num-shards", type=int, default=1, metavar="N",
                   help="split the work N ways for N parallel GPU workers")
    p.add_argument("--shard", type=int, default=0, metavar="K",
                   help="which shard (0-based) this worker handles")
    p.add_argument("--fps", type=float, default=None,
                   help="override the per-experiment sampling rate (see SAMPLING)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="override the per-experiment frame cap. The 10min chess "
                        "clips hold ~500 board positions, so any cap sees a "
                        "fraction of them — that is the degradation being "
                        "measured, but the cap must be reported with any result")
    p.add_argument("--min-frames", type=int, default=None,
                   help="override the per-experiment frame floor")
    p.add_argument("--max-pixels", type=int, default=360 * 420)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override the per-experiment output budget "
                        "(see MAX_NEW_TOKENS)")
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = plain greedy (default). Greedy decoding collapses "
                        "into repetition on ~half of ASL clips: the model loops "
                        "inside GLOSS, exhausts the token budget, and never emits "
                        "TRANSLATION, so the row scores as garbage for reasons "
                        "that have nothing to do with signing. ~1.05-1.1 "
                        "suppresses that. Changing it changes comparability, so "
                        "it is opt-in and recorded per row")
    p.add_argument("--resume", action="store_true",
                   help="skip sample_ids already present in the output")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would run, load no model, write nothing")
    return p.parse_args(argv)


def extract_translation(text: str) -> tuple[str, bool]:
    """Pull the TRANSLATION section out of an ASL response.

    `asl_task.txt` asks for GLOSS / TRANSLATION / CONFIDENCE, but `AslScorer`
    compares `model_output` verbatim against a plain English sentence. Scoring
    the whole response therefore counts the gloss and the confidence note as
    translation text, which inflates WER past 1.0 and drives token_f1 to ~0.
    Only the TRANSLATION body is the hypothesis; the raw reply is kept in
    `model_output_raw` so nothing is discarded.

    Falls back to the full text when no TRANSLATION label is present — a model
    that ignored the format is better scored on what it did say than on "".

    Returns (hypothesis, matched). `matched` is False for the fallback, so the
    caller can tell a real extraction from a reply that never had the section.
    """
    if not text:
        return text, False
    # Tolerate **TRANSLATION:**, "TRANSLATION -", leading bullets, any case.
    m = re.search(r"^[^\w]*\**\s*TRANSLATION\s*\**\s*[:\-]\s*", text,
                  re.IGNORECASE | re.MULTILINE)
    if not m:
        return text.strip(), False
    body = text[m.end():]
    # Stop at the next section header (CONFIDENCE, GLOSS, NOTES, ...).
    stop = re.search(r"^[^\w]*\**\s*(CONFIDENCE|GLOSS|NOTES?|EXPLANATION)\b",
                     body, re.IGNORECASE | re.MULTILINE)
    if stop:
        body = body[:stop.start()]
    return (body.strip().strip("*").strip() or text.strip()), True


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{n}: bad JSON — {exc}") from exc
    if not rows:
        raise SystemExit(f"{path} has no records")
    return rows


def select(rows: list[dict], conditions, stratify, limit) -> list[dict]:
    """Filter, then stratify, then cap — so a pilot spans every condition."""
    if conditions:
        keep = {c.strip() for c in conditions.split(",") if c.strip()}
        unknown = keep - {r["condition"] for r in rows}
        if unknown:
            raise SystemExit(f"no such condition(s): {sorted(unknown)}")
        rows = [r for r in rows if r["condition"] in keep]

    if stratify:
        seen: dict[str, int] = {}
        out = []
        for r in rows:
            c = r["condition"]
            if seen.get(c, 0) < stratify:
                seen[c] = seen.get(c, 0) + 1
                out.append(r)
        rows = out

    if limit is not None:
        rows = rows[:limit]
    return rows


def shard_rows(rows: list[dict], shard: int, num_shards: int) -> list[dict]:
    """Take this worker's slice, balanced by cost.

    Plain `i % n` would be badly skewed: a 10 min chess clip costs ~100x a 5 s
    one, so a worker that happened to collect the long clips would still be
    running long after the others idled. Sorting by duration and dealing
    round-robin (longest-processing-time-first) keeps the slowest items spread
    one-per-worker, which bounds the finish-time spread to roughly one item.
    """
    if num_shards <= 1:
        return rows
    if not 0 <= shard < num_shards:
        raise SystemExit(f"--shard must be in [0,{num_shards}), got {shard}")
    ordered = sorted(rows, key=lambda r: -float(r.get("duration_sec") or 0.0))
    return [r for i, r in enumerate(ordered) if i % num_shards == shard]


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A row whose generation failed is not done; retry it.
            if d.get("model_output"):
                done.add(d["sample_id"])
    return done


def build_messages(row: dict, media_path: Path, args, samp: dict) -> tuple[str | None, list]:
    """(system_prompt, user_content) for one row. Never includes ground truth."""
    from pov import prompts

    experiment = row["experiment"]
    condition = row["condition"]
    instruction = prompts.for_condition(experiment, condition)

    # wbw_mcq splits task (system) from the per-condition user message; chess and
    # ASL put the whole task in one instruction.
    system = prompts.get("wbw_mcq") if experiment == "wbw_mcq" else None

    if row["media_type"] == "image":
        media = {"type": "image", "image": str(media_path),
                 "max_pixels": args.max_pixels}
    else:
        media = {"type": "video", "video": str(media_path),
                 "max_pixels": args.max_pixels, "fps": samp["fps"],
                 "min_frames": samp["min_frames"],
                 "max_frames": samp["max_frames"]}

    return system, [media, {"type": "text", "text": instruction}]


class Runner:
    """Wraps the model. Mirrors the working setup in video_sychophancy."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[load] {model_id} ...", flush=True)
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Thinking models emit <think> blocks that wreck the scorers' parsing.
        self.tkw = {}
        try:
            probe = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
            self.processor.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            self.tkw = {"enable_thinking": False}
            print("[load] thinking disabled", flush=True)
        except TypeError:
            print("[load] enable_thinking unsupported; leaving default", flush=True)
        print(f"[load] done in {time.time() - t0:.0f}s", flush=True)

    def ask(self, system, content, max_new_tokens: int,
            repetition_penalty: float = 1.0) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = []
        if system:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.tkw)
        images, videos, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)
        # Newer processors reject list-valued fps; coerce to a scalar. An
        # image-only message yields an *empty* fps list, so indexing [0]
        # unconditionally breaks every static_image row.
        fps = video_kwargs.get("fps")
        if isinstance(fps, (list, tuple)):
            if fps:
                video_kwargs["fps"] = float(fps[0])
            else:
                video_kwargs.pop("fps")

        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt",
                                **video_kwargs).to("cuda")
        with torch.inference_mode():
            gen = {"max_new_tokens": max_new_tokens, "do_sample": False}
            if repetition_penalty and repetition_penalty != 1.0:
                gen["repetition_penalty"] = repetition_penalty
            out = self.model.generate(**inputs, **gen)
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir
    manifest = args.manifest or run_dir / "manifest.jsonl"
    if args.output:
        output = args.output
    elif args.num_shards > 1:
        output = run_dir / f"preds.shard{args.shard}of{args.num_shards}.jsonl"
    else:
        output = run_dir / "preds.jsonl"

    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest}")

    rows = load_manifest(manifest)
    total = len(rows)
    rows = select(rows, args.conditions, args.stratify, args.limit)
    selected_total = len(rows)
    rows = shard_rows(rows, args.shard, args.num_shards)

    # Verify media exists before loading a 9B model for nothing.
    missing = [r["sample_id"] for r in rows
               if not (run_dir / r["media_path"]).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} media file(s) missing under {run_dir}, "
            f"first: {missing[0]}")

    done = already_done(output) if args.resume else set()
    todo = [r for r in rows if r["sample_id"] not in done]

    by_cond: dict[str, int] = {}
    for r in todo:
        by_cond[r["condition"]] = by_cond.get(r["condition"], 0) + 1

    print(f"manifest   : {manifest}  ({total} rows)")
    shard_note = (f"  |  shard {args.shard}/{args.num_shards} of {selected_total}"
                  if args.num_shards > 1 else "")
    print(f"selected   : {len(rows)}{shard_note}  |  already done: {len(done)}"
          f"  |  to run: {len(todo)}")
    print(f"output     : {output}")
    print(f"model      : {args.model}")
    for exp in sorted({r["experiment"] for r in rows}):
        s = sampling_for(exp, args)
        print(f"sampling   : {exp:8} fps={s['fps']} max_frames={s['max_frames']} "
              f"min_frames={s['min_frames']} max_pixels={args.max_pixels} "
              f"max_new_tokens={max_new_tokens_for(exp, args)} "
              f"rep_penalty={repetition_penalty_for(exp, args)}")
    for c in sorted(by_cond):
        print(f"  {c:24} {by_cond[c]}")

    if args.dry_run:
        print("dry-run: no model loaded, nothing written")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    runner = Runner(args.model)
    mode = "a" if (args.resume and output.exists()) else "w"
    ok = fail = 0
    t0 = time.time()

    with open(output, mode) as fh:
        for i, row in enumerate(todo, 1):
            media_path = run_dir / row["media_path"]
            samp = sampling_for(row["experiment"], args)
            mnt = max_new_tokens_for(row["experiment"], args)
            rpen = repetition_penalty_for(row["experiment"], args)
            system, content = build_messages(row, media_path, args, samp)
            assert not any(f in json.dumps(content) for f in
                           (row.get("ground_truth") or "\x00",)), "ground truth leaked"
            try:
                answer = runner.ask(system, content, mnt,
                                    args.repetition_penalty)
                err = ""
                ok += 1
            except Exception as exc:  # one bad clip must not kill the run
                answer, err = "", f"{type(exc).__name__}: {exc}"
                fail += 1
                print(f"  [{i}/{len(todo)}] {row['sample_id']} FAILED — {err}",
                      flush=True)

            out_row = dict(row)
            # ASL is scored verbatim against a plain English sentence, so the
            # hypothesis must be the TRANSLATION body, not the whole reply.
            if row["experiment"] == "asl" and answer:
                hypothesis, matched = extract_translation(answer)
                out_row["translation_section_found"] = matched
                if matched:
                    out_row["model_output_raw"] = answer
                answer = hypothesis
            out_row["model_output"] = answer
            # Recorded per row: sampling varies by experiment, so a preds file
            # must say how each row's video was actually read.
            out_row.update({
                "model": args.model,
                "sample_fps": samp["fps"],
                "sample_max_frames": samp["max_frames"],
                "sample_min_frames": samp["min_frames"],
                "sample_max_pixels": args.max_pixels,
                "sample_max_new_tokens": mnt,
                "sample_repetition_penalty": rpen,
                "sample_repetition_penalty": args.repetition_penalty,
            })
            if err:
                out_row["predict_error"] = err
            fh.write(json.dumps(out_row) + "\n")
            fh.flush()

            if i % 5 == 0 or i == len(todo):
                el = time.time() - t0
                rate = el / i
                print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} "
                      f"{el:.0f}s elapsed, ~{rate * (len(todo) - i):.0f}s left",
                      flush=True)

    print(f"\nwrote {ok + fail} row(s) to {output}  (ok={ok} failed={fail})")
    print(f"elapsed_sec: {time.time() - t0:.1f}")
    print(f"next: pov eval -i {output}")
    return 1 if fail and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
