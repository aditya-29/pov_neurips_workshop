# POV

Benchmark generator for studying how multimodal models read the **same content**
presented as a single image versus as a video that unrolls over time.

Three experiments, one pipeline:

| Experiment | Content | Video condition | Image / reference condition |
|---|---|---|---|
| `chess` | Synthetic random-legal chess games | Animated board, one clip per target duration | One frame per half-move (static reel) |
| `asl` | How2Sign ASL clips | Signing clip, bucketed by duration | — (video only) |
| `wbw_mcq` | MMLU-style multiple choice | Question revealed one word at a time (`vanishing` / `cumulative`, 3 speeds) | Whole question as one static image |

## Workflow

Generation and evaluation are **separate modules** joined by one CSV.

```
                 configs/chess.yaml
                         │
                         ▼
                  pov generate            →  data/chess/<run_id>/
                                                 ├── media/*.mp4
                                                 ├── ground_truth/*.txt
                                                 ├── config.resolved.yaml
                                                 └── manifest.csv
                         │
                         ▼
              (you run your own model,
               add a `model_output` column)
                         │
                         ▼
                    pov eval              →  scored.csv + summary.csv
```

`pov` never calls a model. Generation writes `manifest.csv`; you fill in
`model_output`; evaluation scores it.

## Install

```bash
pip install -e ".[dev]"
```

`ffmpeg` must be on `PATH` for media generation. Optional chess SVG piece art:
`pip install -e ".[chess-svg]"`.

## Generate

```bash
pov generate -c configs/chess.yaml
pov generate -c configs/wbw_mcq.yaml
pov generate -c configs/asl.yaml
```

## Evaluate

```bash
pov eval -i data/chess/<run_id>/manifest_with_predictions.csv -o results/
```

See `docs/` for the manifest schema and per-experiment configuration reference.
