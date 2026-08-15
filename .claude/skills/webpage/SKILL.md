---
name: webpage
description: Compile pov evaluation results into a single self-contained HTML page with aggregate score tables and a per-sample browser that plays the actual videos inline. Use after `pov eval` has produced scored.jsonl, or when asked to visualise/report/summarise benchmark results as a web page.
---

# webpage

Builds one HTML file from evaluation results: score tables per condition, plus a
searchable, sortable per-sample view where each sample's **actual video plays
inline** next to its ground truth, model output, and scores.

All CSS and JS are inlined — the page needs no network and no build step. Videos
are referenced by **relative path**, not base64-embedded, which keeps the file at
tens of KB instead of the tens of MB the old `viz.py` produced. The page must
therefore stay next to its run directory (see *Placement* below).

## Prerequisites

Scored rows from `pov eval`:

```bash
pov eval -i data/<experiment>/<run_id>/manifest_with_predictions.jsonl
# writes scored.jsonl + summary.jsonl into that run directory
```

If the user has not run eval yet, do that first — the report needs the
`score_*` fields.

## Building the page

Default (writes `scored.html` beside the input):

```bash
pov report -i data/<experiment>/<run_id>/scored.jsonl
```

Explicit output path and title:

```bash
pov report \
  -i data/chess/<run_id>/scored.jsonl \
  -o reports/chess.html \
  --title "Chess transcription — video duration sweep"
```

Options:

- `-o, --output` — where to write the `.html` (default: input path with `.html`)
- `--run-dir` — directory holding `media/`, if the file was moved away from its run
- `--title` — page heading
- `--max-samples N` — cap how many samples are listed (the tables still cover all rows)

You can also produce the page directly from `pov eval`:

```bash
pov eval -i preds.csv --report report.html
```

## Placement — this matters

Media is linked relatively, so the HTML must be able to reach `media/`.

- **Safe:** write the page **inside the run directory** (the default), or anywhere
  that keeps a valid relative path to `media/`.
- **Breaks the videos:** emailing the `.html` on its own, or moving it to a
  different filesystem, leaves every player empty.
- To share, zip the run directory (or at least `report.html` + `media/`) together.

If the user explicitly wants a **portable single file** with playable video, say
that this means base64-embedding and will produce a very large page (the previous
iteration of this project hit 20–41 MB); confirm before doing it, and prefer
`--max-samples` to keep it bounded.

## After building

Report the output path and its size, and mention the page is interactive:
search box, condition filter, and score sort. Do not open a browser unless asked.

## Verifying

A quick check that the page is well-formed and its media links resolve:

```bash
python3 - <<'PY'
import json, re
from pathlib import Path
p = Path("reports/chess.html")
data = json.loads(re.search(r'id="pov-data">(.*?)</script>', p.read_text(), re.S).group(1))
missing = [s["id"] for s in data["samples"] if s["mediaSrc"] and not (p.parent/s["mediaSrc"]).exists()]
print(f"{len(data['samples'])} samples, {len(missing)} broken media links")
PY
```

If links are broken, the fix is `--run-dir`, pointing at the directory that
contains `media/`.

## Programmatic use

```python
from pov.report import build_report_from_file

build_report_from_file(
    "data/chess/run1/scored.jsonl",
    "reports/chess.html",
    title="Chess duration sweep",
)
```

`pov.report.build_report` takes already-loaded rows plus a precomputed summary,
for when a report is assembled in the same process as the evaluation.
