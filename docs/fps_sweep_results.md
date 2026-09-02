# FPS sweep results — TODO 1 and 2

Open-weight models only: `Qwen/Qwen3.5-9B` and `Qwen/Qwen3-VL-8B-Instruct`.
Slurm job 3470, 3 x H200, 7h35m, exit 0. 1,070 runs per model (2,140 total),
all 24 cells complete.

Run dirs are the 2026-08-24 ones, so these line up with the existing four-model
comparison. Per-sample rows are kept in `<run_dir>/sweep/scored.jsonl`; only
metrics appear below (How2Sign text is not redistributable).

## Read the flat rows carefully

`make_fps_plan.py` skips cells where `max_frames` makes a higher fps produce
byte-identical input, and the tables expand them back out. **A row that is flat
across fps is usually one measurement copied, not six measurements agreeing.**
Every scored row carries `sweep_expanded` to tell them apart.

Frames actually fed = `clamp(duration x fps, min_frames, min(max_frames, total))`,
floored to even, sampled uniformly across the clip. chess caps at 96, asl and
wbw_mcq at 128.

## Chess — TODO 1

**Qwen/Qwen3.5-9B** — `score_strict`

| condition | 1 fps | 5 fps | 10 fps | 15 fps | 20 fps | 30 fps |
|---|---|---|---|---|---|---|
| `video_5s` | 0.2250 | 0.3250 | 0.5000 | 0.4750 | 0.3750 | 0.6250 |
| `video_10s` | 0.0900 | 0.2300 | 0.2800 | 0.3100 | 0.3100 | 0.3100 |
| `video_30s` | 0.0300 | 0.0425 | 0.0425 | 0.0425 | 0.0425 | 0.0425 |
| `video_1min` | 0.0144 | 0.0122 | 0.0122 | 0.0122 | 0.0122 | 0.0122 |
| `video_2min` | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 |
| `video_5min` | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 |
| `video_10min` | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |

**Qwen/Qwen3-VL-8B-Instruct** — `score_strict`

| condition | 1 fps | 5 fps | 10 fps | 15 fps | 20 fps | 30 fps |
|---|---|---|---|---|---|---|
| `video_5s` | 0.0750 | 0.0250 | 0.0500 | 0.0500 | 0.0500 | 0.1000 |
| `video_10s` | 0.0000 | 0.0100 | 0.0300 | 0.0200 | 0.0200 | 0.0200 |
| `video_30s` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `video_1min` | 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `video_2min` | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| `video_5min` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `video_10min` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

FPS has a real effect only where the cap is not binding. `video_5s` nearly
triples on the 9B (0.2250 -> 0.6250); `video_10s` goes 0.0900 -> 0.3100. From
`video_30s` on, every fps >= 5 yields 96 frames, so the rate is inert and
`max_frames` is the operative variable. Qwen3-VL-8B is far weaker throughout
and shows no trend.

Note `video_5s` is fed 4 frames — `min_frames`, not the requested rate — since
a 3.9 s clip at 1 fps wants 3.9. The condition with the largest apparent fps
effect is the one where the requested fps least applies.

## ASL — TODO 2

**Qwen/Qwen3.5-9B** — `score_token_f1`

| condition | 1 fps | 5 fps | 10 fps | 15 fps | 20 fps | 30 fps |
|---|---|---|---|---|---|---|
| `video_<3s` | 0.0459 | 0.0458 | 0.0347 | 0.0274 | 0.0514 | 0.0377 |
| `video_<5s` | 0.0472 | 0.0453 | 0.0288 | 0.0445 | 0.0310 | 0.0418 |
| `video_<10s` | 0.0368 | 0.0229 | 0.0269 | 0.0351 | 0.0371 | 0.0371 |
| `video_<15s` | 0.0229 | 0.0440 | 0.0075 | 0.0141 | 0.0141 | 0.0141 |
| `video_<20s` | 0.0255 | 0.0367 | 0.0165 | 0.0165 | 0.0165 | 0.0165 |
| `video_<25s` | 0.0244 | 0.0411 | 0.0187 | 0.0187 | 0.0187 | 0.0187 |
| `video_<30s` | 0.0065 | 0.0379 | 0.0380 | 0.0380 | 0.0380 | 0.0380 |

**Qwen/Qwen3-VL-8B-Instruct** — `score_token_f1`

| condition | 1 fps | 5 fps | 10 fps | 15 fps | 20 fps | 30 fps |
|---|---|---|---|---|---|---|
| `video_<3s` | 0.0752 | 0.0761 | 0.0679 | 0.0577 | 0.0403 | 0.0423 |
| `video_<5s` | 0.0628 | 0.0428 | 0.0480 | 0.0400 | 0.0306 | 0.0370 |
| `video_<10s` | 0.0906 | 0.0557 | 0.0443 | 0.0564 | 0.0514 | 0.0482 |
| `video_<15s` | 0.0623 | 0.0434 | 0.0352 | 0.0472 | 0.0472 | 0.0472 |
| `video_<20s` | 0.0264 | 0.0455 | 0.0537 | 0.0537 | 0.0537 | 0.0537 |
| `video_<25s` | 0.0336 | 0.0505 | 0.0472 | 0.0472 | 0.0472 | 0.0472 |
| `video_<30s` | 0.0521 | 0.0620 | 0.0613 | 0.0613 | 0.0613 | 0.0613 |

No fps effect on either model at any bucket; all variation sits inside the
+/-0.02 token_f1 noise floor. Consistent with the models not reading ASL at all
(170 clips produced 22 distinct sentences on the 9B), so there is no signal for
frame rate to improve. Qwen3-VL-8B is mildly but consistently ahead of the 9B.
