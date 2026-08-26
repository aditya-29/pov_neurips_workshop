#!/usr/bin/env python
"""Fill `model_output` on a pov manifest using a hosted model via Portkey.

The GPU counterpart is scripts/predict_qwen.py; this is the same step-5 bridge
for gateway-hosted models, and it deliberately imports that script's sampling
tables and ASL translation extraction so the two cannot drift apart.

    python scripts/predict_api.py --run-dir data/chess/<id> \
        --model @anthropic-default/claude-opus-5 --tag claude --stratify 20

Every model is sent the *same* frames qwen_vl_utils hands the local models, so
Claude, Gemini and Qwen all score an identical frame set and differences are
attributable to the model rather than to the input. `input_mode` is recorded on
every row anyway, so a scored file always states this rather than implying it.

Requests go through Portkey's OpenAI-compatible chat.completions surface, which
costs the provider-native controls: no thinking/effort configuration, no
server-side refusal fallbacks, and no typed refusal details. A refusal arrives
as an ordinary short completion, so `api_finish_reason` is the only signal.

Ground truth is never sent to a model. `ground_truth`, `source_name`, and
`source_path` are read only to be copied through to the output.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict_qwen import (  # noqa: E402  - shared so sampling cannot drift apart
    MAX_NEW_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    SAMPLING,
    DEFAULT_SAMPLING,
    extract_translation,
    load_manifest,
    select,
)

# Model identity is the gateway slug the user's Portkey exposes; there is no
# safe default, so --model is required rather than guessed.


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--model", required=True,
                   help="Portkey model slug, e.g. "
                        "@anthropic-default/claude-opus-5")
    p.add_argument("--tag", required=True,
                   help="output filename suffix, e.g. claude or gemini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stratify", type=int, default=None, metavar="N",
                   help="N rows per condition — the way to build a subset that "
                        "still covers every condition")
    p.add_argument("--conditions", default=None)
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel in-flight requests. These are IO-bound, so "
                        "this is unrelated to GPU count")
    p.add_argument("--max-frames", type=int, default=None,
                   help="override the per-experiment frame cap. Each frame is a "
                        "separate image in the request, so this drives both cost "
                        "and any per-request image limit the provider enforces")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--token-param", default="max_tokens",
                   choices=("max_tokens", "max_completion_tokens"),
                   help="which output-budget field the provider accepts. The "
                        "newer OpenAI reasoning models reject max_tokens with a "
                        "400 and require max_completion_tokens")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan, estimate frames, call nothing")
    return p.parse_args(argv)


def sampling_for(experiment: str, args) -> dict:
    s = dict(SAMPLING.get(experiment, DEFAULT_SAMPLING))
    if args.fps is not None:
        s["fps"] = args.fps
    if args.max_frames is not None:
        s["max_frames"] = args.max_frames
    return s


def max_new_tokens_for(experiment: str, args) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    return MAX_NEW_TOKENS.get(experiment, DEFAULT_MAX_NEW_TOKENS)


def build_prompt(row: dict) -> tuple[str | None, str]:
    """(system, user_text). Never contains ground truth."""
    from pov import prompts
    experiment = row["experiment"]
    instruction = prompts.for_condition(experiment, row["condition"])
    system = prompts.get("wbw_mcq") if experiment == "wbw_mcq" else None
    return system, instruction


def extract_frames(media_path: Path, samp: dict, quality: int) -> list[bytes]:
    """The exact frames qwen_vl_utils would hand a local model, as JPEG bytes.

    Reusing process_vision_info rather than decoding independently is the whole
    point: it keeps Claude's input identical to what the Qwen runs scored.
    """
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import numpy as np

    msg = [{"role": "user", "content": [{
        "type": "video", "video": str(media_path),
        "max_pixels": 360 * 420, "fps": samp["fps"],
        "min_frames": samp["min_frames"], "max_frames": samp["max_frames"],
    }]}]
    _, videos, _ = process_vision_info(msg, return_video_kwargs=True)
    vid = videos[0]  # (T, C, H, W)
    out = []
    for frame in vid:
        arr = frame.permute(1, 2, 0).to("cpu").numpy()
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        out.append(buf.getvalue())
    return out


def encode_image_file(path: Path, quality: int) -> bytes:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class PortkeyRunner:
    """One gateway, many providers, OpenAI-compatible chat.completions shape.

    Every model gets the same sampled frames, which is why `input_mode` is
    constant here: Claude, Gemini and the local Qwen runs all score the identical
    frame set, so the comparison is about the model and nothing else.

    Going through the gateway costs the provider-native controls -- no
    `output_config.effort`, no adaptive-thinking configuration, no server-side
    refusal fallbacks, and no typed `stop_details`. A refusal arrives as an
    ordinary short completion, so it is inferred from `finish_reason` rather than
    read from a field.
    """

    input_mode = "sampled_frames"

    def __init__(self, model: str, token_param: str = "max_tokens"):
        import os
        from portkey_ai import Portkey
        key = os.environ.get("PORTKEY_API_KEY")
        if not key:
            raise SystemExit("PORTKEY_API_KEY is not set in the environment")
        self.client = Portkey(api_key=key)
        self.model = model
        self.token_param = token_param

    def ask(self, row, media_path, samp, max_tokens, quality) -> tuple[str, dict]:
        system, text = build_prompt(row)
        if row["media_type"] == "image":
            frames = [encode_image_file(media_path, quality)]
        else:
            frames = extract_frames(media_path, samp, quality)

        parts = [{"type": "image_url",
                  "image_url": {"url": "data:image/jpeg;base64," +
                                base64.standard_b64encode(f).decode()}}
                 for f in frames]
        parts.append({"type": "text", "text": text})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": parts})

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            **{self.token_param: max_tokens},
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        meta = {"n_frames": len(frames), "finish_reason": choice.finish_reason}
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["input_tokens"] = getattr(usage, "prompt_tokens", None)
            meta["output_tokens"] = getattr(usage, "completion_tokens", None)
        return answer, meta


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("model_output"):
            done.add(d["sample_id"])
    return done


def main(argv=None) -> int:
    args = parse_args(argv)
    model = args.model
    tag = args.tag
    run_dir = args.run_dir
    manifest = run_dir / "manifest.jsonl"
    output = run_dir / f"preds.{tag}.jsonl"

    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest}")
    rows = load_manifest(manifest)
    total = len(rows)
    rows = select(rows, args.conditions, args.stratify, args.limit)

    missing = [r["sample_id"] for r in rows
               if not (run_dir / r["media_path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} media file(s) missing, first {missing[0]}")

    done = already_done(output) if args.resume else set()
    todo = [r for r in rows if r["sample_id"] not in done]

    est_frames = 0
    for r in todo:
        s = sampling_for(r["experiment"], args)
        if r["media_type"] == "image":
            est_frames += 1
        else:
            est_frames += max(s["min_frames"],
                              min(s["max_frames"], int(r["duration_sec"] * s["fps"])))

    print(f"manifest   : {manifest} ({total} rows)")
    print(f"selected   : {len(rows)}  |  done: {len(done)}  |  to run: {len(todo)}")
    print(f"model      : {model}")
    print(f"output     : {output}")
    print(f"concurrency: {args.concurrency}   token_param: {args.token_param}")
    if True:
        print(f"input      : sampled frames  (~{est_frames} images, "
              f"~{est_frames * 176 / 1e6:.2f}M image tokens at ~176 tok/frame)")
        over = sum(1 for r in todo if r["media_type"] != "image"
                   and max(sampling_for(r['experiment'], args)["min_frames"],
                           min(sampling_for(r['experiment'], args)["max_frames"],
                               int(r["duration_sec"] * sampling_for(r['experiment'], args)["fps"]))) > 100)
        if over:
            print(f"  NOTE: {over} row(s) exceed 100 frames/request; if the API "
                  f"caps images per request, cap them with --max-frames")
    for exp in sorted({r["experiment"] for r in todo}):
        print(f"  {exp:9} max_new_tokens={max_new_tokens_for(exp, args)}")

    if args.dry_run:
        print("dry-run: nothing called, nothing written")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    runner = PortkeyRunner(model, args.token_param)

    lock = threading.Lock()
    fh = open(output, "a" if (args.resume and output.exists()) else "w")
    counts = {"ok": 0, "fail": 0}
    t0 = time.time()

    def work(row):
        media_path = run_dir / row["media_path"]
        samp = sampling_for(row["experiment"], args)
        mnt = max_new_tokens_for(row["experiment"], args)
        try:
            answer, meta = runner.ask(row, media_path, samp, mnt, args.jpeg_quality)
            err = ""
        except Exception as exc:
            answer, meta, err = "", {}, f"{type(exc).__name__}: {exc}"

        out = dict(row)
        if row["experiment"] == "asl" and answer:
            hyp, matched = extract_translation(answer)
            out["translation_section_found"] = matched
            if matched:
                out["model_output_raw"] = answer
            answer = hyp
        out["model_output"] = answer
        out.update({
            "model": model,
            "provider": "portkey",
            "input_mode": runner.input_mode,
            "sample_fps": samp["fps"],
            "sample_max_frames": samp["max_frames"],
            "sample_max_new_tokens": mnt,
        })
        out.update({f"api_{k}": v for k, v in meta.items()})
        if err:
            out["predict_error"] = err

        with lock:
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            counts["ok" if not err else "fail"] += 1
            n = counts["ok"] + counts["fail"]
            if err:
                print(f"  [{n}/{len(todo)}] {row['sample_id']} FAILED — {err}",
                      flush=True)
            if n % 5 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  [{n}/{len(todo)}] ok={counts['ok']} fail={counts['fail']} "
                      f"{el:.0f}s, ~{el / n * (len(todo) - n):.0f}s left", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(work, r) for r in todo]
            for f in as_completed(futs):
                f.result()
    finally:
        fh.close()

    # --resume appends, so a retried row would otherwise appear twice and be
    # double-counted by `pov eval`. Collapse on sample_id, and never let an
    # empty retry overwrite an answer an earlier pass already got.
    kept = {}
    with open(output) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r["sample_id"]
            if sid in kept and not (r.get("model_output") or "").strip():
                continue
            kept[sid] = r
    with open(output, "w") as fh:
        for r in kept.values():
            fh.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(kept)} unique row(s) to {output} "
          f"(this pass: ok={counts['ok']} failed={counts['fail']})")
    print(f"elapsed_sec: {time.time() - t0:.1f}")
    print(f"next: pov eval -i {output} -o {run_dir}/eval/{tag}")
    return 1 if counts["fail"] and not counts["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
