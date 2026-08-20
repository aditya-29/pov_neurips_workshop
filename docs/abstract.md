# Workshop submission — title and abstract

**Status:** draft for an abstract-only submission (~250 words).
**Predecessor:** Sethuraman T V et al., *Stress Tests REVEAL Fragile Temporal and
Visual Grounding in Video-Language Models*, arXiv:2602.11244.

---

## Title

> **From REVEAL to TRACE — Attributing Fragile Temporal Grounding in Video-Language Models**

**TRACE** — *Temporal Reveal-rate Analysis of Cross-frame Evidence*.

The name is also the verb of the programme: REVEAL established *that* temporal
grounding is fragile; TRACE follows that fragility back to *which* mechanism
breaks.

---

## Abstract

REVEAL showed that video-language models under-use the visual evidence in front
of them. We ask why, and find they are not so much failing to see as declining to
look twice: they seek a shortcut, locating a single frame that makes the question
answerable and answering from that frame alone. When a benchmark forecloses that
shortcut, accuracy collapses — for frontier proprietary and open-weight models
alike.

Two failure modes hide behind one score. Evidence may never be **delivered** to
the model, discarded by frame sampling and context budgets before inference
begins; or it may be delivered and never **integrated** across frames. Reported as
a single number the two are indistinguishable, and the second is invisible.

TRACE separates them by construction, by controlling the schedule on which
content is revealed. The same semantic content is re-rendered under presentation
conditions differing *only* in how that content is distributed over time. In our
word-by-word task, one condition presents an entire multiple-choice question as a
single image; a second accumulates words, so the final frame still holds the whole
question; a third shows one word per frame, so no single frame is ever sufficient.
Comparable accuracy on the first two and collapse on the third is direct evidence
of shortcut-seeking rather than temporal reading. Sweeping reveal rate and clip
duration — over synthetic chess games (5 s–10 min), continuous ASL translation,
and MMLU — yields regimes in which every token is provably sampled and the clip
fits comfortably in context, isolating cross-frame integration, not perception or
context length, as the binding constraint.

TRACE ships as a reproducible generator: configs, media, per-item ground truth,
and scoring.

*(~265 words. If hard-capped at 250, cut the final sentence first.)*

---

## Notes for revision

### Optional numeric slot

The abstract states the finding qualitatively and contains **no fabricated
numbers**. If you want magnitudes, the natural insertion point is after
*"accuracy collapses"*:

> "…accuracy collapses by N points…"

Two numbers carry the whole argument:

1. **The cumulative → vanishing gap.** Evidence of shortcut-seeking (the
   integration gate). Both conditions contain identical content; only `cumulative`
   leaves a single sufficient frame.
2. **The share of failures explained by unsampled frames.** The delivery gate.
   Computable from `frames_per_word`, `n_frames` and `duration_sec` in the
   manifest, against each model's sampling rate.

### Naming the models

Currently "frontier proprietary and open-weight models". Naming them (Gemini,
Qwen-VL) costs four words and buys credibility — reviewers weight this.

### Lineage phrasing

Opening with *"REVEAL showed…"* assumes familiarity. At a venue where the group is
known, that is an asset. Elsewhere, consider *"Prior work has shown that
video-language models under-use the visual evidence in front of them [cite]"* —
same lineage via citation, no assumed familiarity.

### Claim discipline

The defensible claim is **not** "models do not understand video". It is the
narrower, harder-to-attack one the design actually measures: **they succeed when a
single frame suffices and fail when it does not.**

### Design point held in reserve

The chess frames caption only the move number (`Move 8`) — no piece, no squares,
no algebraic notation. An earlier version printed the move in the caption, which
let a model score by OCR alone. Mentioning that the textual shortcut was ablated
pre-empts an obvious reviewer objection, if there is room.

---

## Title alternatives considered

Kept here in case the framing shifts.

| Title | Emphasis |
|---|---|
| TRACE Locates the Source of the Fragile Temporal Grounding REVEAL Exposed in Video-Language Models | most complete, self-explaining |
| TRACE Explains What REVEAL Exposed in Video-Language Models | shortest, most quotable |
| TRACE Attributes the Fragile Temporal Grounding REVEAL Exposed to Sampling and Integration Failures | only one naming both gates |
| TRACE Separates Unseen from Unintegrated Evidence REVEAL Could Not Distinguish | sharpest contribution statement |
| Timing Tests TRACE Fragile Temporal Grounding to Its Source in Video-Language Models | closest structural mirror of REVEAL |

Other benchmark names considered: GLANCE (*Gauging Language models' Aggregation of
Non-Concurrent Evidence*), PACE (*Paced Assessment of Cross-frame Evidence*), DRIP
(*Distributed Reveal for Integration Probing*), SIFT (*Sampling Isolated From
Temporal-integration*).

---

## How the codebase backs each claim

| Abstract claim | Where it comes from |
|---|---|
| "a single frame that makes the question answerable" | `wbw_mcq` `cumulative` mode: frame *i* shows `words[:i+1]`, so the final frame contains the entire question |
| "no single frame is ever sufficient" | `wbw_mcq` `vanishing` mode: each frame holds exactly one word (verified: max 1) |
| "identical content, differing only in temporal distribution" | all three conditions derive from the same `Question`; `static_image`, `vanishing_*`, `cumulative_*` share `question_id` and `ground_truth` |
| "every token is provably sampled" | `speed_wps` = 0.5 → `frames_per_word` = 60 → ~2 s per word at 30 fps; `frames_per_word` is recorded per row |
| "the clip fits comfortably in context" | `n_frames` and `duration_sec` are recorded per row, read back from the file with `ffprobe` |
| "synthetic chess games (5 s–10 min)" | `configs/chess.yaml` durations, 7 clips per game, all prefixes of one game |
| "continuous ASL translation" | `configs/asl.yaml`, How2Sign val, duration buckets `<3s`…`<30s` |
| "reproducible generator" | `pov generate -c <config>`; every run writes `config.resolved.yaml` and a `config_hash` on each manifest row |
