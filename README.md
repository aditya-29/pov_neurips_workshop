# POV

Benchmark generator for studying how multimodal models read the **same content**
presented as a single image versus as a video that unrolls over time.

Three experiments, one YAML-driven pipeline:

| Experiment | Content | Video condition | Image / reference condition |
|---|---|---|---|
| `chess` | Synthetic random-legal chess games | Animated board, one clip per target duration (5 s → 10 min) | — |
| `asl` | How2Sign ASL clips | Signing clip, sampled evenly across duration buckets | — |
| `wbw_mcq` | MMLU-style multiple choice | Question revealed one word at a time (`vanishing` / `cumulative`, any number of speeds) | Whole question as one static image |

## Workflow

Generation and evaluation are **separate modules**, joined by one CSV. `pov`
never calls a model.

```
        configs/chess.yaml
                │
                ▼
         pov generate  ──────►  data/chess/<run_id>/
                                    ├── media/                 the videos / images
                                    ├── ground_truth/          full transcripts
                                    ├── config.resolved.yaml   exactly what produced this
                                    └── manifest.jsonl           one row per artifact
                │
                ▼
    you run your own model and fill in
      the empty `model_output` column
                │
                ▼
           pov eval  ──────────►  scored.jsonl   per-row scores
                                  summary.jsonl  means per condition
                │
                ▼
          pov report  ─────────►  report.html  one file, videos playable inline
```

## Install

```bash
pip install -e ".[dev]"
```

`ffmpeg` (with `ffprobe`) must be on `PATH` for media generation:
`brew install ffmpeg` / `apt-get install ffmpeg`.

Optional: `pip install -e ".[chess-svg]"` uses python-chess SVG piece art. Without
it the board falls back to Unicode chess glyphs, then to lettered discs — all
three render at the same size, so nothing else changes.

## 0. Get the source data

`chess` is fully synthetic — it needs nothing. The other two read real corpora,
which are **not** bundled (How2Sign is CC BY-NC and not redistributable):

```bash
pip install -e ".[data]"

python scripts/fetch_mmlu.py --limit 200      # → data/questions.jsonl   (small)
./scripts/download_how2sign.sh                # → data/asl_source/       (~1.7 GB)
```

| Experiment | Source | Acquired by | Lands in |
|---|---|---|---|
| `chess` | generated in-process | — | — |
| `wbw_mcq` | [MMLU](https://huggingface.co/datasets/cais/mmlu) (`cais/mmlu`, MIT) | `scripts/fetch_mmlu.py` | `data/questions.jsonl` |
| `asl` | [How2Sign](https://how2sign.github.io/) val, sentence-level RGB front clips + translation CSV (CC BY-NC 4.0, research use only) | `scripts/download_how2sign.sh` | `data/asl_source/` |

`fetch_mmlu.py` samples evenly across MMLU subjects with a fixed seed, skips
questions over `--max-words` (a 300-word stem makes an unwatchable video), and
can convert a local MMLU CSV release offline with `--from-csv`.
`download_how2sign.sh` takes `--split val|test|train` and `--dest`, resumes, and
verifies what it downloaded is really How2Sign rather than a Drive error page.

Both destinations are gitignored. `examples/questions.jsonl` holds 5 hand-written
questions for a no-download smoke test — it is *not* real MMLU.

## 1. Generate

```bash
pov generate -c configs/wbw_mcq.yaml
pov generate -c configs/chess.yaml
pov generate -c configs/asl.yaml          # edit the paths in it first
```

Useful flags:

```bash
pov generate -c configs/chess.yaml --dry-run              # validate + show the plan
pov generate -c configs/chess.yaml --set run.workers=16   # override any value
pov validate -c configs/chess.yaml                        # config check only
```

`--set` takes `dotted.key=value`, parsed as YAML (`--set video.tune=null`). Nested
**mappings merge** rather than replace, so `--set params.speeds.fast=8` changes one
speed and leaves the others; to change the *set* of speeds, edit the YAML.

Every run writes `config.resolved.yaml` and stamps a `config_hash` on each
manifest row, so a row always points back at what produced it. Runs are keyed by
`run_id` (a timestamp plus that hash unless you name it), so nothing is silently
overwritten. Re-running with `resume: true` skips artifacts that already exist.

## 2. Predict

Fill the `model_output` column — one row is one media file plus its ground truth.

```python
import pandas as pd

df = pd.read_json("data/chess/<run_id>/manifest.jsonl", lines=True)
run_dir = "data/chess/<run_id>"

for i, row in df.iterrows():
    df.at[i, "model_output"] = my_model(f"{run_dir}/{row.media_path}")

df.to_json(f"{run_dir}/preds.jsonl", orient="records", lines=True)
```

`pov` itself uses only the stdlib, so pandas is **not** installed with it —
`pip install pandas` first, or read the file with one `json.loads` per line.
`pov eval` accepts CSV too, if that is easier to produce.

Comparing several models? Add a `model` column and stack the rows — `pov eval`
groups by it automatically.

## 3. Evaluate

```bash
pov eval -i data/chess/<run_id>/preds.jsonl
```

| Experiment | Metrics |
|---|---|
| `chess` | `strict` (longest correct prefix), `loose` (LCS), `hybrid` (only runs of ≥2 consecutive correct moves), plus move counts. Matched on (colour, from-square, to-square) — piece names are parsed but ignored, since models often name the piece wrongly while reading the squares correctly. |
| `wbw_mcq` | `correct`, `answered`, `refusal`. Answers are read from `ANSWER: X`, then a leading letter, then `(X)`, then a standalone A–D. A refusal with no letter is flagged rather than scored wrong. |
| `asl` | `exact_match`, `token_f1`, smoothed `bleu`, `char_similarity`, `wer`. Articles are stripped before comparison because ASL omits them. Optional `judge_strict` / `judge_loose` columns you supply are aggregated alongside. |

Scores land in `score_*` columns so they can never collide with a manifest column.

## 4. Report

```bash
pov report -i data/chess/<run_id>/scored.jsonl
```

One self-contained HTML file: score tables per condition, plus a searchable,
sortable sample browser where each video plays inline next to its ground truth,
model output, and scores. All CSS/JS is inlined and no network is needed.

Media is referenced by **relative path**, not base64-embedded, so the page stays
tens of KB rather than tens of MB. **Keep the HTML with its run directory** — zip
them together to share. See the `/webpage` skill for details.

## Layout

```
pov/
  config.py       YAML loading, validation, hashing
  layout.py       run directory structure
  manifest.py     the CSV contract
  video.py        ffmpeg pipe writer, probing, clip cutting
  registry.py     experiment name → generator
  cli.py
  experiments/    generation (never imported by eval)
    chess/        engine.py · render.py · generate.py
    wbw_mcq/      question.py · render.py · generate.py
    asl/          sampling.py · generate.py
  eval/           scoring (imports nothing from experiments, calls no model)
  report/         single-file HTML
configs/          one example config per experiment
docs/manifest.md  every manifest column
tests/            591 tests
```

## Speed

Video generation was the bottleneck in the previous iteration; three changes
removed most of it.

1. **Frames are piped straight to ffmpeg.** The old pipeline wrote every frame to
   disk as a JPEG and handed ffmpeg the directory.
2. **Held frames are never re-rendered or re-piped.** `write_timeline` takes
   `(frame, n_output_frames)` segments and divides all hold counts by their GCD,
   piping the reduced set at `fps/gcd` and letting ffmpeg restore the timing. A
   60-frame hold is piped once. Output is frame-exact (asserted against `ffprobe`).
3. **Frames are shared across conditions.** One chess game renders its positions
   once for *every* duration, and one question renders its word frames once for
   *every* speed — speed only changes hold length.

Measured: a 3-game × 3-duration chess run renders 63 frames to encode 3,861
(-88% pipe writes); word-by-word MCQ at three speeds renders 94 frames to encode
8,004 (-75%).

`run.workers` parallelises across games / questions / clips.

## Tests

```bash
pytest                  # 591 tests, ~9s
pytest -m integration   # only the ones that write real media
pytest -m "not slow"    # skip perft(4) and the download-script tests
```

The pure-python core (chess rules, config validation, manifest, bucketing, all
metrics) has no heavy dependencies and is tested offline. Tests that encode real
MP4s are marked `integration` and skip automatically without ffmpeg. Chess move
generation is pinned by `perft(1..4)` against the published node counts
(20 / 400 / 8,902 / 197,281).

## Contributing

`.claude/skills/git-commit/SKILL.md` defines the commit discipline for this repo:
commit any pending work before starting a change, then commit the change itself,
with subjects of the form `claude - <code-change>`.
