# POV

Benchmark generator for studying how multimodal models read the **same content**
presented as a single image versus as a video that unrolls over time.

Three experiments, one YAML-driven pipeline:

| Experiment | Content | What varies | Ground truth |
|---|---|---|---|
| `chess` | Synthetic random-legal chess games | Clip duration: 5 s → 10 min of the same game | The move transcript |
| `asl` | How2Sign ASL clips | Clip duration bucket: `<3s` → `<30s` | The English sentence |
| `wbw_mcq` | MMLU-style multiple choice | Whole question as one image **vs.** revealed one word at a time at 3 speeds | The answer letter `A`–`D` |

---

# Read this first (especially if you are an agent)

These are the invariants. Everything below is consistent with them.

1. **`pov` never calls a model.** It generates media and it scores predictions.
   Running the model is your job, in between. There is no API key, no network
   call, no inference code anywhere in the package.
2. **Generation and evaluation are joined by one file:** `manifest.jsonl`. You add
   a `model_output` value to each record and hand the same file back.
3. **The manifest is JSONL** — one JSON object per line, values keep real types
   (`n_frames` is an `int`; absent values are `null`). `pov eval` also accepts CSV.
4. **`media_path` is relative to the run directory**, never absolute. Resolve it as
   `<run_dir>/<media_path>`.
5. **Do not invent field names, condition values, or CLI flags.** Complete lists
   are given below. If something is not listed here, it does not exist.
6. **`chess` needs no source data.** `asl` and `wbw_mcq` do — see §1.
7. Every command in this file has been executed and its output verified.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the run completed but some items failed (it prints an `ERRORS` block) |
| `2` | bad input: invalid config, missing source data, unusable file |

An error starting `internal error:` is a bug in pov, not a problem with your input.

---

# 0. Install

```bash
pip install -e ".[dev]"
```

`ffmpeg` **and** `ffprobe` must be on `PATH` — generation cannot work without them:

```bash
brew install ffmpeg          # macOS
sudo apt-get install ffmpeg  # Debian/Ubuntu
ffmpeg -version              # verify
```

Optional extras:

| Extra | Install | Effect |
|---|---|---|
| `data` | `pip install -e ".[data]"` | `datasets` + `gdown`, needed only by `scripts/` to fetch source corpora |
| `chess-svg` | `pip install -e ".[chess-svg]"` | python-chess SVG piece art. Without it the board falls back to Unicode glyphs — generation still works, and the run prints a note |

`pov` itself needs only `pyyaml`, `numpy`, `pillow`, `tqdm`.

---

# 1. Get the source data

| Experiment | Needs source data? | How |
|---|---|---|
| `chess` | **No** — fully synthetic | nothing to do |
| `wbw_mcq` | Yes — MMLU questions | `python scripts/fetch_mmlu.py --limit 200` |
| `asl` | Yes — How2Sign videos | `./scripts/download_how2sign.sh` |

## wbw_mcq — MMLU

```bash
pip install -e ".[data]"
python scripts/fetch_mmlu.py --limit 200        # → data/questions.jsonl
```

Downloads `cais/mmlu` from HuggingFace, samples evenly across subjects with a
fixed seed, and skips questions over `--max-words` (default 120 — a 300-word stem
makes an unwatchable video).

**`--limit` is applied after downloading all 58 subjects** (~3 min). To fetch only
some: `--subjects anatomy astronomy world_religions`.

Offline alternative, if you have the original MMLU CSV release:
`python scripts/fetch_mmlu.py --from-csv /path/to/mmlu/test`

`examples/questions.jsonl` holds **5 hand-written questions** for a no-download
smoke test. It is **not** real MMLU — do not report results from it.

## asl — How2Sign

```bash
pip install -e ".[data]"
./scripts/download_how2sign.sh                  # → data/asl_source/  (~1.7 GB, ~35 s)
```

Downloads the **sentence-level Green Screen RGB clips (frontal)** plus the English
translation CSV from <https://how2sign.github.io/>. Flags: `--split val|test|train`
(default `val`), `--dest DIR`, `--yes` (required for the 31 GB train split).

Licence: **CC BY-NC 4.0, research use only, not redistributable.**

Already have How2Sign? Skip the download and point the config at it:

```bash
pov generate -c configs/asl.yaml \
  --set params.metadata_csv=/full/path/to/how2sign_val.csv \
  --set params.video_dir=/full/path/to/raw_videos
```

Your source videos are only ever **read**. pov writes normalised copies into the
run directory and never modifies or deletes the originals.

---

# 2. Generate

```bash
pov generate -c configs/chess.yaml
pov generate -c configs/wbw_mcq.yaml
pov generate -c configs/asl.yaml
```

Always check first with `--dry-run`. It validates the config **and** the presence
of source data, and writes nothing:

```bash
pov generate -c configs/asl.yaml --dry-run
```

```
experiment : asl
run id     : 20260817-203302-f07e2e8fc64a
run dir    : data/asl/20260817-203302-f07e2e8fc64a
manifest   : data/asl/20260817-203302-f07e2e8fc64a/manifest.jsonl
config hash: f07e2e8fc64a
encoder    : libx264 crf=20 preset=veryfast fps=30
source     : 1739 clips in /path/to/raw_videos
config is valid, source data is present; no media written (--dry-run)
```

If source data is missing it exits **2** and names the fix. A failed run writes
**nothing at all** — no empty directories are left behind.

## Overriding config values

`--set dotted.key=value`, repeatable. Values parse as YAML.

```bash
pov generate -c configs/chess.yaml \
  --set run.workers=16 \
  --set params.num_games=5 \
  --set run.run_id=pilot \
  --set video.tune=null \
  --set "params.durations=[{label: 10s, seconds: 10}]"
```

**Mappings merge key-by-key; lists replace wholesale.** So
`--set params.speeds.fast=8` changes one speed and leaves the others. Even
`--set "params.speeds={normal: 2.0}"` *merges* — to genuinely reduce the set of
speeds, edit the YAML.

## What each experiment produces

| Experiment | Shipped config produces | Time |
|---|---|---|
| `chess` | 20 games × 7 durations = **140 clips**, ~123 MB | ~63 s |
| `asl` | up to 28 per bucket × 7 buckets = **≤196 clips**, ~183 MB | ~22 s |
| `wbw_mcq` | 200 questions × 7 conditions = **1,400 files** | ~10 min |

`asl` yields fewer than 196 because the longest buckets have fewer qualifying
videos; short buckets are **not** topped up, since that would skew the duration
distribution the experiment exists to vary.

`wbw_mcq` is slow because the `slow` speed is 0.5 words/sec = **60 frames per
word** (~70 % of all encoded frames). Dropping it cuts the run to ~3 min.

## Resuming

`resume: true` (default) skips any artifact already on disk, so an interrupted run
continues where it stopped. `overwrite: true` deletes the run directory first.
They are mutually exclusive — setting both is a config error.

---

# 3. Where the output lives

```
data/<experiment>/<run_id>/
├── media/                  ← the videos and images
├── ground_truth/           ← ONLY for chess (see below)
├── config.resolved.yaml    ← exactly what produced this run
└── manifest.jsonl          ← one record per artifact  ← START HERE
```

`run_id` defaults to `<timestamp>-<config_hash>`, e.g.
`20260817-203302-f07e2e8fc64a`. Set `run.run_id` to name it yourself.

Real example:

```
data/asl/20260817-203302-f07e2e8fc64a/
├── media/          170 × .mp4, 854×480, 183 MB
├── manifest.jsonl  170 records
└── config.resolved.yaml
```

## Where is the ground truth?

**In the manifest**, in the `ground_truth` field of every record. That is the
answer key for all three experiments.

`ground_truth/` as a *directory* exists **only when the ground truth is too long
for one field** — in practice **chess only**, whose transcripts run to thousands of
characters. When it exists, `ground_truth_path` points into it.

| Experiment | `ground_truth` field | `ground_truth_path` | `ground_truth/` dir |
|---|---|---|---|
| `chess` | compact `1 White b2 b3; 1 Black c7 c5; …` (truncated if long) | set, e.g. `ground_truth/game0000_5s.txt` | **yes** |
| `asl` | the full English sentence | empty | no |
| `wbw_mcq` | the answer letter, `A`–`D` | empty | no |

**Rule:** if `ground_truth_path` is non-empty, read that file for the authoritative
value; otherwise use the `ground_truth` field.
`pov.manifest.load_ground_truth(row, run_dir)` does exactly this.

## Where is the media?

`<run_dir>/<media_path>`. `media_path` is always relative and always inside
`media/`.

```python
from pathlib import Path
run_dir = Path("data/asl/20260817-203302-f07e2e8fc64a")
clip = run_dir / row["media_path"]      # media/00dWJ4YRRSI_7-1-rgb_front.mp4
```

---

# 4. The manifest

One JSON object per line. Field order is stable: core fields, then
experiment-specific fields, then `cfg_*`, then `model_output` last.

## Core fields — all 20, present for every experiment

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | str | Unique id. Also the media filename stem |
| `experiment` | str | `chess` \| `asl` \| `wbw_mcq`. Selects the scorer |
| `condition` | str | The presentation condition — main grouping key (values below) |
| `media_type` | str | `video` or `image` |
| `media_path` | str | Path to the media, **relative to the run directory** |
| `media_filename` | str | Basename of `media_path` |
| `ground_truth` | str | The answer key |
| `ground_truth_path` | str | Relative path to the full ground truth, or `""` |
| `fps` | float\|null | Frames per second (`null` for images) |
| `n_frames` | int | Frame count (`1` for images) |
| `duration_sec` | float\|null | Duration (`null` for images) |
| `width` | int | Pixels |
| `height` | int | Pixels |
| `codec` | str | `h264` for video, `jpeg` for images |
| `file_size_bytes` | int | Size on disk |
| `seed` | int | Seed that produced this sample |
| `run_id` | str | The run this belongs to |
| `config_hash` | str | Hash of content-affecting config. Equal hashes = same settings |
| `pov_version` | str | Version that generated it |
| `generated_at` | str | ISO timestamp |
| `model_output` | str | **Empty when written. You fill this in** |

`fps`, `n_frames`, `duration_sec`, `width`, `height` and `codec` are read back from
the finished file with `ffprobe` — they describe what was really produced, not what
was requested.

## Experiment-specific fields

**chess** (7): `game_index`, `duration_label`, `target_seconds`, `n_half_moves`,
`game_total_half_moves`, `game_result`, `motion`

- `n_half_moves` — half-moves shown in *this* clip; also caps the ground truth during scoring
- `game_index` — clips sharing this are prefixes of the same game
- `target_seconds` vs `duration_sec` — requested vs achieved; clips hold whole moves only, so short ones can fall ~20 % short

**wbw_mcq** (14): `question_id`, `mode`, `speed`, `speed_wps`, `frames_per_word`,
`word_count`, `stem`, `option_a`, `option_b`, `option_c`, `option_d`, `domain`,
`difficulty`, `question_source`

- `question_id` — rows sharing this are the same question in different conditions
- `speed`, `speed_wps`, `frames_per_word` are `null` for the static image

**asl** (6): `bucket`, `source_name`, `source_duration_sec`, `source_path`,
`video_id`, `sentence_id`

## Condition values — these are the complete lists

| Experiment | Conditions |
|---|---|
| `chess` | `video_<label>` per configured duration: `video_5s`, `video_10s`, `video_30s`, `video_1min`, `video_2min`, `video_5min`, `video_10min` |
| `asl` | `video_<bucket>`: `video_<3s`, `video_<5s`, `video_<10s`, `video_<15s`, `video_<20s`, `video_<25s`, `video_<30s` |
| `wbw_mcq` | `static_image`, plus `vanishing_<speed>` and `cumulative_<speed>` per configured speed — with shipped defaults: `vanishing_slow`, `vanishing_normal`, `vanishing_fast`, `cumulative_slow`, `cumulative_normal`, `cumulative_fast` |

Labels and speed names come from the config, so `video_3s` or `vanishing_glacial`
are possible if configured. **Never hard-code the list — read it from the manifest.**

## `cfg_*` fields

Every record carries the whole resolved config flattened, e.g. `cfg_video.fps`,
`cfg_params.num_games`. Lists are JSON strings. Identical on every row; they exist
so a record is self-describing when pulled out of its directory.

## Reading it

```python
import json
rows = [json.loads(l) for l in open("data/asl/<run_id>/manifest.jsonl")]

# or
import pandas as pd
df = pd.read_json("data/asl/<run_id>/manifest.jsonl", lines=True)

# or, handles both JSONL and CSV and validates:
from pov.manifest import read_manifest
rows = read_manifest("data/asl/<run_id>/manifest.jsonl")
```

---

# 5. Predict

## Get the task prompt

The prompts ship with pov so everyone runs the same benchmark:

```bash
pov prompt --list                              # every prompt and its fields
pov prompt chess                               # chess transcription task
pov prompt asl                                 # ASL translation task
pov prompt asl --kind judge                    # LLM-judge rubric
pov prompt wbw_mcq                             # MCQ system prompt
pov prompt wbw_mcq --condition vanishing_slow  # user message for a condition
```

Available: `asl/task`, `asl/judge`, `chess/task`, `chess/task_legacy`,
`wbw_mcq/task`, `wbw_mcq/user_text`, `wbw_mcq/user_static_image`,
`wbw_mcq/user_video`.

```python
from pov import prompts

prompts.get("asl")                                   # task instruction
prompts.for_condition("wbw_mcq", row["condition"])   # correct message per condition
prompts.render("asl", "judge",                       # fills {ground_truth}/{model_output}
               ground_truth=..., model_output=...)
```

`chess/task_legacy` is the original study's wording. It describes a caption that
**no longer exists** (frames now show only `Move N`). Use it only to reproduce old
runs.

## Fill in `model_output`

One record = one media file + its ground truth. Put your model's raw response in
`model_output`.

```python
import json
from pathlib import Path
from pov import prompts

run_dir = Path("data/asl/20260817-203302-f07e2e8fc64a")
rows = [json.loads(l) for l in open(run_dir / "manifest.jsonl")]

for row in rows:
    media = run_dir / row["media_path"]
    instruction = prompts.for_condition(row["experiment"], row["condition"])
    row["model_output"] = my_model(instruction, media)   # your code

with open(run_dir / "preds.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
```

**Do not** send `ground_truth`, `source_name`, or `source_path` to the model —
they contain or point at the answer.

Comparing several models? Add a `model` field and stack the rows into one file;
`pov eval` groups by it automatically.

---

# 6. Evaluate

```bash
pov eval -i data/asl/<run_id>/preds.jsonl
```

Writes `scored.jsonl` (every input field + `score_*` fields) and `summary.jsonl`
(means grouped by experiment, condition, and `model`/`run_id` when present) next to
the input. JSONL in → JSONL out; CSV in → CSV out.

Flags: `-o/--output-dir`, `--run-dir` (where `ground_truth_path` resolves from,
default the input's directory), `--group-by a,b`, `--report out.html`.

## Metrics — complete list

| Experiment | Fields added (prefixed `score_`) |
|---|---|
| `chess` | `strict`, `loose`, `hybrid`, `moves_matched`, `moves_expected`, `moves_predicted` |
| `asl` | `exact_match`, `token_f1`, `bleu`, `char_similarity`, `wer` |
| `wbw_mcq` | `correct`, `answered`, `refusal`, plus `predicted_answer` (text, not averaged) |

- **chess** — `strict` = longest correct prefix; `loose` = longest common
  subsequence; `hybrid` = only runs of ≥2 consecutive correct moves. By
  construction `strict ≤ hybrid ≤ loose`. Matched on
  **(colour, from-square, to-square)**; piece names are parsed but ignored, because
  models often name the piece wrongly while reading the squares correctly.
- **asl** — articles are stripped before comparison because ASL omits them.
  `wer` is an **error** rate: lower is better, and it can exceed 1.0.
- **wbw_mcq** — answers read from `ANSWER: X`, then a leading letter, then `(X)`,
  then a standalone `A`–`D`. A refusal with no letter is flagged, not scored wrong.

An **empty `model_output` is scored as a wrong answer, not skipped** — so `wer`
becomes 1.0, and `moves_expected` still reflects the ground truth.

## Optional LLM judge for ASL

`pov` never calls a judge. If you run one yourself (use `pov prompt asl --kind judge`,
which emits `{"strict": bool, "loose": bool, "explanation": str}`), put the values
in `judge_strict` / `judge_loose` fields and `pov eval` aggregates them alongside
the deterministic metrics. Accepts `1/0`, `true/false`, `yes/no`, or a float.

---

# 7. Report

```bash
pov report -i data/asl/<run_id>/scored.jsonl
```

One self-contained HTML file: score tables per condition, plus a searchable,
sortable sample browser with each video playing inline next to its ground truth,
model output, and scores. All CSS/JS inlined, no network needed.

Media is referenced by **relative path**, not embedded, so the page stays tens of
KB. **Keep the HTML with its run directory** — zip them together to share, or the
players will be empty. If you move it, pass `--run-dir`.

Flags: `-o/--output`, `--run-dir`, `--title`, `--max-samples N`.

---

# 8. Config reference

Every key, with the default used when omitted. Unknown keys are a **hard error** —
a typo like `fsp: 30` will never silently fall back to a default.

## `run:` — all experiments

| Key | Default | Notes |
|---|---|---|
| `output_root` | `data` | Root for all runs |
| `run_id` | `null` | `null` → `<timestamp>-<config_hash>` |
| `seed` | `1234` | Same seed reproduces the run exactly |
| `workers` | `8` | Parallelism. Does not affect output content |
| `overwrite` | `false` | Deletes the run directory first |
| `resume` | `true` | Skips existing artifacts. Cannot both be true with `overwrite` |

## `video:` — all experiments

| Key | Default | Notes |
|---|---|---|
| `fps` | `30` | |
| `crf` | `18` | 0 lossless … 51 worst |
| `preset` | `veryfast` | `ultrafast`…`veryslow` |
| `tune` | `stillimage` | `null` for real footage (ASL uses `null`) |
| `pix_fmt` | `yuv420p` | Requires **even** frame dimensions |
| `codec` | `libx264` | |
| `threads` | `0` | 0 = ffmpeg decides |

## `params:` — chess

| Key | Default | Notes |
|---|---|---|
| `num_games` | `10` (shipped config: `20`) | |
| `durations` | *required* | List of `{label, seconds}` |
| `motion` | `static` | `static` = one frame per half-move. `animated` slides pieces and is far slower |
| `max_half_moves` | `400` (shipped: `600`) | Must exceed what the longest duration needs |
| `max_game_attempts` | `12` | Retries with new seeds until a game is long enough |
| `timing.intro_frames` | `60` | Capped by `intro_share` |
| `timing.hold_frames` | `36` | Frames each position is held (`static`) |
| `timing.slide_frames` | `14` | `animated` only |
| `timing.pause_frames` | `22` | `animated` only |
| `timing.outro_frames` | `120` | Capped by `outro_share` |
| `timing.intro_share` | `0.10` | Max share of the clip the intro may take |
| `timing.outro_share` | `0.20` | Keeps short clips honest |
| `theme.square` | `48` | 48 → 416×480 frames |
| `theme.padding` | `16` | |
| `theme.panel` | `64` | Bottom label strip; `0` removes it |
| `show_labels` | `true` | Draws `Move N` **only** — never the move itself |
| `use_svg_pieces` | `true` | Falls back to glyphs without `[chess-svg]` |

## `params:` — wbw_mcq

| Key | Default | Notes |
|---|---|---|
| `questions_path` | *required* | `.jsonl` or MMLU `.csv` |
| `questions_format` | `auto` | `auto` \| `jsonl` \| `mmlu_csv` |
| `limit` | `null` | Cap the number of questions |
| `shuffle` | `false` | Shuffles before applying `limit`, using `run.seed` |
| `static_image` | `true` | The image condition |
| `modes` | `[vanishing, cumulative]` | |
| `speeds` | `{slow: 0.5, normal: 2.0, fast: 5.0}` | Words per second. Names become filenames |
| `blank_gap_frames` | `4` | Blank frames between words (`vanishing`) |
| `min_frames_per_word` | `1` | |
| `jpeg_quality` | `90` | |
| `canvas.width` / `.height` | `1200` / `800` | Both must be **even** |
| `canvas.video_font_size` | `36` | |
| `canvas.video_padding` | `80` | |
| `canvas.video_line_gap` | `8` | |
| `canvas.static_font_size` | `18` | |
| `canvas.static_padding` | `60` | |
| `canvas.static_line_gap` | `14` | |

Accepted question formats: `{stem, options:{A..D}, answer:"B"}`, HuggingFace
`{question, choices:[...], answer:1}`, flat `A`/`B`/`C`/`D` keys, or headerless
MMLU CSV `question,A,B,C,D,answer`. `answer` may be a letter or an index.

## `params:` — asl

| Key | Default | Notes |
|---|---|---|
| `metadata_csv` | *required* | How2Sign translation CSV |
| `video_dir` | *required* | Directory of source `.mp4` |
| `delimiter` | `auto` | How2Sign is tab-separated |
| `id_column` | `SENTENCE_NAME` | Also becomes `sample_id` |
| `text_column` | `SENTENCE` | Becomes `ground_truth` |
| `duration_column` | `null` | Set to reuse durations in the CSV and skip probing |
| `video_extension` | `.mp4` | Must start with a dot |
| `buckets.edges` | *required* | Half-open `[low, high)` |
| `buckets.labels` | auto | e.g. `<3s`, `<2min` |
| `samples_per_bucket` | `null` | **Exactly one** of this or `total_samples` |
| `total_samples` | `null` | Split across buckets, remainder to the earliest |
| `scale_height` | `480` | Width follows aspect ratio, forced even. `null` keeps source |
| `target_fps` | `null` | `null` uses `video.fps` |
| `max_clip_seconds` | `null` | Truncate long clips |
| `use_time_range` | `false` | `true` cuts `START`..`END` from each source |
| `start_column` / `end_column` | `START` / `END` | |
| `probe_workers` | `16` | Parallel `ffprobe` calls |
| `duration_cache` | `auto` | `auto` → `<output_root>/asl/.pov_durations.json`, or a path, or `none` |

---

# 9. CLI reference

```
pov generate  -c CONFIG [--set KEY=VALUE ...] [--dry-run] [--quiet]
pov eval      -i INPUT [-o DIR] [--run-dir DIR] [--group-by COLS] [--report HTML]
pov report    -i INPUT [-o OUT.html] [--run-dir DIR] [--title T] [--max-samples N]
pov validate  -c CONFIG [--set KEY=VALUE ...]
pov prompt    [EXPERIMENT] [-k KIND] [--condition COND] [--list]
pov experiments
pov --version
```

That is the complete surface. There is no `pov run`, no `pov predict`, no
`pov download` — those do not exist.

---

# 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `metadata_csv not found: /how2sign_val.csv` | A shell variable was unset and expanded to empty. Use the full path |
| `cannot generate — source data is not ready` | Run the fetch script in §1, or point the config at an existing copy |
| `questions file not found: data/questions.jsonl` | Run `python scripts/fetch_mmlu.py --limit 200` |
| `ffmpeg not found on PATH` | Install ffmpeg |
| `config error: params: unknown key(s) [...]` | A typo. The message lists the valid keys |
| `frame size 1201x800 must have even dimensions` | `yuv420p` needs even width and height |
| Generation seems to hang | It is encoding. `wbw_mcq` at defaults is ~1.75 M frames / ~10 min. A progress bar shows on a terminal |
| Report videos are blank | The HTML was moved away from its run directory. Keep them together or pass `--run-dir` |
| `no scorer for experiment 'x'` | The `experiment` field must be `chess`, `asl`, or `wbw_mcq` |

---

# 11. Tests

```bash
pytest                  # 748 tests, ~11 s
pytest -m integration   # only those writing real media
pytest -m "not slow"    # skip perft(4)
```

The pure-python core (chess rules, config validation, manifest, bucketing, all
metrics, prompts) has no heavy dependencies and runs offline. Tests that encode
real MP4s are marked `integration` and skip automatically without ffmpeg. Chess
move generation is pinned by `perft(1..4)` against the published node counts
(20 / 400 / 8,902 / 197,281).

**A clean clone must pass everything.** `tests/test_selfcontained.py` enforces
it: every file under `pov/` is tracked by git, no ignore rule hides one, no
source hardcodes a path outside the repo, and nothing imports a package that
`pyproject.toml` does not declare. This repo has no dependency on any other
checkout.

Four tests skip by default — they compare the vendored prompts byte-for-byte
against the original study, which is not part of this repo. To run them, point
at a copy of it:

```bash
POV_REFERENCE_REPO=/path/to/icml_workshop pytest tests/test_prompts.py
```

---

# 12. Repository layout

```
pov/
  config.py       YAML loading, validation, hashing
  layout.py       run directory structure
  manifest.py     the JSONL contract
  video.py        ffmpeg pipe writer, probing, clip cutting
  progress.py     progress bars
  errors.py       PovError — every deliberate error derives from it
  registry.py     experiment name → generator
  cli.py
  experiments/    generation (never imported by eval)
    chess/        engine.py · render.py · generate.py
    wbw_mcq/      question.py · render.py · generate.py
    asl/          sampling.py · generate.py
  eval/           scoring (imports nothing from experiments, calls no model)
  prompts/        task instructions + data/*.txt
  report/         single-file HTML
configs/          one config per experiment
scripts/          fetch_mmlu.py · download_how2sign.sh
.github/          CI: full matrix + a clean-wheel-install job
docs/manifest.md  the manifest schema in depth
examples/         questions.jsonl (5 demo questions, NOT real MMLU)
tests/            748 tests
```

## Speed

Video generation was the original bottleneck; three changes removed most of it:

1. **Frames pipe straight into ffmpeg** — the old pipeline wrote every frame to
   disk as a JPEG first.
2. **Held frames are never re-rendered or re-piped.** `write_timeline` divides all
   hold counts by their GCD and lets ffmpeg restore the timing, so a 60-frame hold
   is piped once. Output is frame-exact, asserted against `ffprobe`.
3. **Frames are shared across conditions** — one chess game renders its positions
   once for every duration; one question renders its word frames once for every
   speed.

Measured: chess renders 63 frames to encode 3,861 (−88 % pipe writes); word-by-word
MCQ renders 94 to encode 8,004 (−75 %).

## Continuous integration

`.github/workflows/tests.yml` runs the suite on Python 3.10–3.13, and a separate
`clean-install` job builds a wheel, installs it *outside* the source tree with
only the declared runtime dependencies, and runs generate → eval → report. That
second job is the one that catches packaging mistakes: a prompt missing from the
wheel fails there while the normal matrix stays green.

## Results website

The interactive evaluation atlas lives in [`website/`](website/). It has no
build step or package dependencies. From the repository root, run:

```bash
python3 -m http.server 8000 --directory website
```

Then open `http://localhost:8000`.

## Experiment TODOs

Unless noted otherwise, **FPS means the frame-sampling rate supplied to the
model**, not the encoded FPS of the source video. Keep the generated media,
prompts, decoding settings, scoring code, and matched sample IDs fixed within
each sweep so that only the declared variable changes.

- [ ] **Chess FPS sweep**
  - Evaluate every generated chess video with every model.
  - Sweep input sampling FPS over **1, 5, 10, 15, 20, and 30 FPS**.
  - Report every chess metric by model, video-duration condition, and FPS.
  - Preserve per-sample results so FPS effects can be compared on identical
    games rather than only through aggregate scores.

- [ ] **ASL translation FPS sweep**
  - Evaluate the complete ASL evaluation set with every model.
  - Sweep input sampling FPS over **1, 5, 10, 15, 20, and 30 FPS**.
  - Report every translation metric by model, duration bucket, and FPS.
  - Use the same clips and matched sample IDs at every FPS. Continue to follow
    the How2Sign licence and redaction requirements.

- [ ] **Word-by-word MCQ factorial sweep**
  - Evaluate every generated word-by-word MCQ video with every model.
  - Run the full Cartesian product of:
    - input sampling FPS: **1, 5, 10, 15, 20, 30**;
    - target video length: **5, 10, 15, 20, 30 seconds**;
    - presentation speed: **slow, normal, fast**.
  - Define speeds numerically and keep them fixed across video lengths:
    **slow = 0.5 words/second**, **normal = 2 words/second**, and
    **fast = 5 words/second**, matching `configs/wbw_mcq.yaml`.
  - Treat this as a **6 × 5 × 3 factorial matrix (90 conditions per model)**,
    evaluated separately for cumulative and vanishing presentation modes.
  - Specify a deterministic length policy before generation: clips shorter
    than the target require padding, while clips longer than the target require
    a documented truncation or time-normalization rule.
  - Report accuracy by model, presentation mode, FPS, target length, and speed;
    retain per-sample results and coverage statistics for every matrix cell.

## Licence

The code is MIT — see `LICENSE`. **That does not cover the datasets.** How2Sign
is CC BY-NC 4.0 (research use only, non-commercial, not redistributable) and
MMLU carries its own upstream terms; generated media inherits them. Do not
commit or republish clips derived from either.

## Contributing

`.claude/skills/git-commit/SKILL.md` defines the commit discipline: commit pending
work before starting a change, then commit the change, with subjects of the form
`claude - <code-change>`.
