#!/usr/bin/env python
"""Plan an FPS sweep that skips byte-identical requests.

`max_frames` caps the frames handed to the model, so past a certain fps every
higher rate yields the same frame count and therefore the same request. On the
long chess clips all six rates collapse to one. Running them anyway would spend
six times the compute for identical inputs and, worse, would report six equal
numbers as if fps had been varied and found not to matter.

For each item this records the lowest fps producing each distinct frame count,
and writes one id-list per fps. Cells that duplicate a lower fps are omitted;
`plan.json` records the mapping so the analysis can expand them back out.
"""
import json, sys
from pathlib import Path

FPS = [1, 5, 10, 15, 20, 30]
RUNS = {
    "chess":   ("data/chess/20260824-001859-547f861a109a", 4, 96),
    "asl":     ("data/asl/20260824-002004-e1225b96852f", 8, 128),
}


def frames(dur, fps, mn, mx):
    return max(mn, min(mx, int(dur * fps)))


def main(outdir="data/fps_sweep_plan"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {}
    for exp, (base, mn, mx) in RUNS.items():
        rows = [json.loads(l) for l in open(Path(base) / "manifest.jsonl")]
        per_fps = {f: [] for f in FPS}
        alias = {}          # sample_id -> {fps: canonical fps it duplicates}
        for r in rows:
            sid, dur = r["sample_id"], r["duration_sec"]
            seen = {}       # frame count -> first fps that produced it
            for f in FPS:
                n = frames(dur, f, mn, mx)
                if n in seen:
                    alias.setdefault(sid, {})[f] = seen[n]
                else:
                    seen[n] = f
                    per_fps[f].append(sid)
        for f in FPS:
            p = out / f"{exp}_fps{f}.ids"
            p.write_text("\n".join(per_fps[f]) + "\n")
        plan[exp] = {
            "base": base, "min_frames": mn, "max_frames": mx,
            "counts": {str(f): len(per_fps[f]) for f in FPS},
            "n_items": len(rows),
            "alias": alias,
        }
        tot = sum(len(v) for v in per_fps.values())
        print(f"{exp}: {len(rows)} items, {len(rows)*len(FPS)} naive -> {tot} runs "
              f"({100*(1-tot/(len(rows)*len(FPS))):.0f}% skipped as duplicates)")
        print("   per fps: " + ", ".join(f"{f}:{len(per_fps[f])}" for f in FPS))
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"wrote {out}/plan.json and per-fps id lists")


if __name__ == "__main__":
    sys.exit(main())
