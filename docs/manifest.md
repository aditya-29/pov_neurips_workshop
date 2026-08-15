# The manifest CSV

`manifest.csv` is the only contract between generation and evaluation. One row
describes one media artifact: where it is, what the right answer is, what the
media actually turned out to be, and what config produced it.

Column order is stable: **core columns**, then **experiment-specific columns**
(alphabetical), then **`cfg_*` config columns** (alphabetical), then
**`model_output`** last.

To evaluate, fill in `model_output` and hand the same file to `pov eval`.

## Core columns

Present for every experiment.

| Column | Meaning |
|---|---|
| `sample_id` | Unique id for this artifact. Also the media filename stem. |
| `experiment` | `chess` \| `asl` \| `wbw_mcq`. Selects the scorer. |
| `condition` | The presentation condition, e.g. `video_30s`, `static_image`, `vanishing_slow`. This is the main grouping key in reports. |
| `media_type` | `video` or `image`. |
| `media_path` | Path to the media, **relative to the run directory**. |
| `media_filename` | Basename of `media_path`, for convenience. |
| `ground_truth` | The expected answer, inline. Truncated with a marker when very long (chess transcripts); the full text is then at `ground_truth_path`. |
| `ground_truth_path` | Relative path to the full ground truth file, or empty when it is fully inline. |
| `fps` | Frames per second of the written media (empty for images). |
| `n_frames` | Frame count (1 for images). |
| `duration_sec` | Duration in seconds (empty for images). |
| `width`, `height` | Pixel dimensions. |
| `codec` | e.g. `h264`, `jpeg`. |
| `file_size_bytes` | Size on disk. |
| `seed` | The seed that produced this sample. |
| `run_id` | The run this row belongs to. |
| `config_hash` | Hash of the content-affecting config. Identical hashes mean identical generation settings. |
| `pov_version` | Version of pov that generated the row. |
| `generated_at` | ISO timestamp of the run. |
| `model_output` | **Empty when written.** Put your model's prediction here. |

`fps`, `n_frames`, `duration_sec`, `width`, `height` and `codec` are read back
from the finished file with `ffprobe`, not assumed from the config — they describe
what was really produced, including on resumed runs.

## Experiment-specific columns

### chess

| Column | Meaning |
|---|---|
| `game_index` | Which synthetic game this clip came from. Clips sharing an index are prefixes of one game. |
| `duration_label` | The configured label, e.g. `30s`. |
| `target_seconds` | The requested duration. Compare with `duration_sec` for what was achievable in whole moves. |
| `n_half_moves` | Half-moves shown in this clip. Also caps the ground truth during scoring. |
| `game_total_half_moves` | Length of the full underlying game. |
| `game_result` | Outcome, or `Clip ends at move N` for a prefix. |
| `motion` | `static` (one frame per half-move) or `animated` (sliding pieces). |

Ground truth is the move transcript. The `ground_truth` cell holds a compact
`1 White b2 b3; 1 Black c7 c5; …` form; `ground_truth_path` points at the
human-readable table. The scorer accepts either.

### wbw_mcq

| Column | Meaning |
|---|---|
| `question_id` | Question this artifact belongs to. Rows sharing it are the same question in different conditions. |
| `mode` | `static`, `vanishing`, or `cumulative`. |
| `speed`, `speed_wps` | Speed name and words per second (empty for the static image). |
| `frames_per_word` | Output frames each word is held for. |
| `word_count` | Number of revealed tokens. |
| `stem` | Question text. |
| `option_a` … `option_d` | The four options. |
| `domain`, `difficulty`, `question_source` | Metadata carried from the source dataset. |

Ground truth is the answer letter (`A`–`D`).

### asl

| Column | Meaning |
|---|---|
| `bucket` | Duration bucket the source video fell into, e.g. `<10s`. |
| `source_name` | Original `SENTENCE_NAME`. |
| `source_duration_sec` | Measured duration of the source video (the value used for bucketing). |
| `source_path` | Absolute path to the source video. |
| `video_id`, `sentence_id` | How2Sign identifiers, when present. |

Ground truth is the reference English sentence.

## Config columns

Every row carries the whole resolved config flattened into `cfg_*` columns —
`cfg_video.fps`, `cfg_params.num_games`, and so on. Lists are JSON-encoded. These
are identical on every row of a run; they exist so a row pulled out of its
directory is still self-describing. `config.resolved.yaml` in the run directory is
the readable copy.

## Columns you may add

| Column | Effect |
|---|---|
| `model_output` | Required. The prediction to score. |
| `model` | Optional. When present, `pov eval` groups the summary by it too, so several models can share one file. |
| `judge_strict`, `judge_loose` | Optional, ASL only. Boolean-ish values from your own judge; aggregated alongside the deterministic metrics. Accepts `1/0`, `true/false`, `yes/no`, or a float. |

## Notes

- Media paths are relative, so a run directory can be moved or shared without
  invalidating it.
- Ground truth may contain commas, quotes and newlines; it is properly CSV-quoted.
  Read it with a real CSV parser, never by splitting on commas.
- Chess transcripts can exceed Python's default 128 KiB CSV field limit.
  `pov.manifest.read_manifest` raises that limit for you.
- Empty numeric cells mean *not applicable* (e.g. `fps` on an image), never zero.
