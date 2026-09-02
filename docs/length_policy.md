# wbw_mcq target-length policy (TODO 3)

Aditya's TODO requires a deterministic length policy fixed *before* generation.
Decision (2026-08-30): **pad with blank frames.**

- Words play at the declared rate — slow 0.5 w/s, normal 2 w/s, fast 5 w/s —
  and any remaining time to the target length is blank frames.
- This keeps words/second honest across all five target lengths, so `speed`
  stays the declared variable instead of being silently confounded with
  `target_length`. The cost is clips that end in dead air.
- Padding placement: trailing, after the last word. (Leading pad would give the
  model a run-up that differs by cell.)

## Still undecided — the truncation half

Padding only covers questions *shorter* than the target. A question whose words
do not fit at the declared speed still needs a documented rule: drop the
question from that cell, or truncate the stem. This must be settled before
generation, since it changes which questions appear in which cells and
therefore whether sample IDs stay matched across the matrix.
