"""Single-file HTML report for evaluation results.

Produces one self-contained `.html` — all CSS and JS inline, no CDN, no build
step — that shows aggregate scores per condition and a browsable per-sample
view with the actual videos playing inline.

Videos are referenced by **relative path** rather than base64-embedded: the
original viz.py inlined everything and produced 20–41 MB pages that choke a
browser. The report is written next to (or above) the run directory and links
down into `media/`, so keep the HTML with the data when sharing it.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from pov.eval.base import SCORE_PREFIX

#: Columns never shown in the per-sample detail panel (noise or duplicated).
_HIDDEN_DETAIL_COLUMNS = {
    "media_path", "media_filename", "ground_truth_path", "pov_version",
    "generated_at", "config_hash",
}

_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def build_report(
    rows: Sequence[Mapping],
    output_path: str | Path,
    *,
    run_dir: str | Path,
    summary: Sequence[Mapping] = (),
    metrics: Sequence[str] = (),
    group_columns: Sequence[str] = (),
    title: str = "POV evaluation report",
    max_samples: int | None = None,
) -> Path:
    """Write the report and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)

    samples = [
        _sample_payload(row, run_dir, output_path.parent, metrics) for row in rows
    ]
    if max_samples is not None:
        samples = samples[:max_samples]

    payload = {
        "title": title,
        "metrics": list(metrics),
        "metricLabels": [_short(metric) for metric in metrics],
        "groupColumns": list(group_columns),
        "summary": [dict(record) for record in summary],
        "samples": samples,
    }

    document = _TEMPLATE.format(
        title=html.escape(title),
        style=_STYLE,
        script=_SCRIPT,
        data=_safe_json(payload),
        header=_render_header(rows, samples, metrics),
        summary_table=_render_summary(summary, group_columns, metrics),
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def build_report_from_csv(
    scored_csv: str | Path,
    output_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    title: str | None = None,
    max_samples: int | None = None,
) -> Path:
    """Build a report straight from a scored.csv written by `pov eval`."""
    from pov.eval.runner import summarise
    from pov.manifest import read_manifest

    scored_csv = Path(scored_csv)
    rows = read_manifest(scored_csv)
    if not rows:
        raise ValueError(f"{scored_csv}: no rows")

    base_dir = Path(run_dir) if run_dir is not None else scored_csv.parent
    metrics = [
        column for column in rows[0]
        if column.startswith(SCORE_PREFIX) and _mostly_numeric(rows, column)
    ]
    group_columns = [
        column for column in ("experiment", "condition", "model")
        if column in rows[0]
    ]
    summary = summarise(rows, group_columns, metrics)

    return build_report(
        rows,
        output_path,
        run_dir=base_dir,
        summary=summary,
        metrics=metrics,
        group_columns=group_columns,
        title=title or f"POV evaluation — {scored_csv.parent.name}",
        max_samples=max_samples,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Payload construction
# ──────────────────────────────────────────────────────────────────────────────


def _sample_payload(
    row: Mapping, run_dir: Path, html_dir: Path, metrics: Sequence[str]
) -> dict:
    media_rel = (row.get("media_path") or "").strip()
    media_src = ""
    media_kind = ""
    if media_rel:
        absolute = (run_dir / media_rel).resolve()
        try:
            media_src = os.path.relpath(absolute, html_dir.resolve()).replace(os.sep, "/")
        except ValueError:  # different drive on Windows
            media_src = absolute.as_uri()
        suffix = absolute.suffix.lower()
        if suffix in _VIDEO_EXTENSIONS:
            media_kind = "video"
        elif suffix in _IMAGE_EXTENSIONS:
            media_kind = "image"

    scores = {}
    for metric in metrics:
        value = row.get(metric, "")
        if value not in ("", None):
            try:
                scores[_short(metric)] = float(value)
            except (TypeError, ValueError):
                pass

    details = {
        key: value
        for key, value in row.items()
        if value not in ("", None)
        and key not in _HIDDEN_DETAIL_COLUMNS
        and not key.startswith("cfg_")
        and not key.startswith(SCORE_PREFIX)
        and key not in ("ground_truth", "model_output")
    }

    return {
        "id": row.get("sample_id", ""),
        "experiment": row.get("experiment", ""),
        "condition": row.get("condition", ""),
        "model": row.get("model", ""),
        "mediaSrc": media_src,
        "mediaKind": media_kind,
        "groundTruth": row.get("ground_truth", "") or "",
        "modelOutput": row.get("model_output", "") or "",
        "scores": scores,
        "details": details,
    }


def _render_header(rows: Sequence[Mapping], samples: Sequence[Mapping],
                   metrics: Sequence[str]) -> str:
    experiments = sorted({(row.get("experiment") or "") for row in rows} - {""})
    conditions = sorted({(row.get("condition") or "") for row in rows} - {""})
    with_media = sum(1 for sample in samples if sample["mediaSrc"])

    cards = [
        ("samples", str(len(rows))),
        ("experiments", ", ".join(experiments) or "—"),
        ("conditions", str(len(conditions))),
        ("media linked", str(with_media)),
    ]
    return "\n".join(
        f'<div class="stat"><span class="stat-value">{html.escape(value)}</span>'
        f'<span class="stat-label">{html.escape(label)}</span></div>'
        for label, value in cards
    )


def _render_summary(summary: Sequence[Mapping], group_columns: Sequence[str],
                    metrics: Sequence[str]) -> str:
    if not summary:
        return '<p class="empty">No summary rows.</p>'

    columns = [*group_columns, "n", *metrics]
    numeric_columns = {"n", *metrics}
    head_cells = []
    for column in columns:
        css_class = ' class="num"' if column in numeric_columns else ""
        head_cells.append(f"<th{css_class}>{html.escape(_short(column))}</th>")
    head = "".join(head_cells)

    # Per-metric maxima drive the inline bars, so each column is self-scaling.
    maxima: dict[str, float] = {}
    for metric in metrics:
        values = [_as_float(record.get(metric)) for record in summary]
        values = [value for value in values if value is not None]
        maxima[metric] = max(values) if values else 0.0

    body_rows = []
    for record in summary:
        cells = []
        for column in columns:
            value = record.get(column, "")
            if column in metrics:
                number = _as_float(value)
                if number is None:
                    cells.append('<td class="num">—</td>')
                    continue
                peak = maxima.get(column) or 0.0
                width = (number / peak * 100.0) if peak > 0 else 0.0
                cells.append(
                    f'<td class="num"><span class="bar" style="width:{width:.1f}%"></span>'
                    f"<span class='bar-value'>{number:.4f}</span></td>"
                )
            elif column == "n":
                cells.append(f'<td class="num">{html.escape(str(value))}</td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="table-scroll"><table class="summary">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _safe_json(payload: Any) -> str:
    """JSON for embedding inside a <script> block.

    Model output is untrusted text. A response containing the literal
    ``</script>`` would otherwise close the block early, breaking the page and
    letting the rest of the string be parsed as markup. Escaping `<`, `>` and
    `&` as \\uXXXX makes that impossible; JSON.parse turns them back into the
    original characters, so nothing is lost. U+2028/U+2029 are escaped too —
    they are line terminators to a JavaScript parser.
    """
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _short(column: str) -> str:
    return column[len(SCORE_PREFIX):] if column.startswith(SCORE_PREFIX) else column


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mostly_numeric(rows: Sequence[Mapping], column: str) -> bool:
    seen = 0
    for row in rows:
        value = row.get(column, "")
        if value in ("", None):
            continue
        if _as_float(value) is None:
            return False
        seen += 1
    return seen > 0


# ──────────────────────────────────────────────────────────────────────────────
# Template
# ──────────────────────────────────────────────────────────────────────────────

_STYLE = """
:root {
  --bg: #ffffff; --fg: #16181d; --muted: #666e7a; --line: #e3e6ea;
  --panel: #f7f8fa; --accent: #3b6fd4; --accent-soft: #dbe6fb;
  --good: #1f9254; --bad: #c0392b;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e8eaed; --muted: #9aa3ad; --line: #2a2e35;
    --panel: #1b1e24; --accent: #6f9df0; --accent-soft: #24344f;
    --good: #4ac47f; --bad: #f0776a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; letter-spacing: -.01em; }
.sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }
.stats { display: flex; flex-wrap: wrap; gap: .75rem; margin-bottom: 1rem; }
.stat {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: .6rem .9rem; min-width: 110px;
}
.stat-value { display: block; font-size: 1.15rem; font-weight: 600; }
.stat-label { display: block; color: var(--muted); font-size: .72rem;
  text-transform: uppercase; letter-spacing: .06em; }
.table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; }
th, td { padding: .5rem .7rem; text-align: left; border-bottom: 1px solid var(--line); }
th { background: var(--panel); font-weight: 600; white-space: nowrap;
  position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums;
  position: relative; white-space: nowrap; }
.bar { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent-soft);
  z-index: 0; }
.bar-value { position: relative; z-index: 1; }
.controls { display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1rem; }
input[type=search], select {
  background: var(--bg); color: var(--fg); border: 1px solid var(--line);
  border-radius: 8px; padding: .45rem .6rem; font: inherit; font-size: .86rem;
}
input[type=search] { flex: 1 1 240px; }
.card {
  border: 1px solid var(--line); border-radius: 12px; margin-bottom: .75rem;
  overflow: hidden; background: var(--panel);
}
.card > summary {
  cursor: pointer; padding: .7rem .9rem; display: flex; gap: .6rem;
  align-items: center; flex-wrap: wrap; list-style: none;
}
.card > summary::-webkit-details-marker { display: none; }
.card > summary::before { content: "▸"; color: var(--muted); }
.card[open] > summary::before { content: "▾"; }
.sid { font-family: var(--mono); font-size: .8rem; }
.tag {
  background: var(--accent-soft); color: var(--accent); border-radius: 999px;
  padding: .1rem .55rem; font-size: .72rem; font-weight: 600;
}
.score-chip { margin-left: auto; font-family: var(--mono); font-size: .78rem;
  color: var(--muted); }
.body { padding: 0 .9rem .9rem; display: grid; gap: 1rem;
  grid-template-columns: minmax(0, 1fr); }
@media (min-width: 820px) { .body.has-media { grid-template-columns: 380px minmax(0, 1fr); } }
video, .body img { width: 100%; border-radius: 8px; background: #000; display: block; }
.field { margin-bottom: .7rem; }
.field-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin-bottom: .2rem; }
.field-value { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: .5rem .65rem; font-family: var(--mono); font-size: .8rem;
  white-space: pre-wrap; word-break: break-word; max-height: 260px; overflow: auto; }
.kv { display: flex; flex-wrap: wrap; gap: .35rem; }
.kv span { background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: .1rem .45rem; font-size: .74rem; color: var(--muted); font-family: var(--mono); }
.missing { color: var(--muted); font-style: italic; }
.empty { color: var(--muted); }
footer { margin-top: 3rem; color: var(--muted); font-size: .78rem; }
"""

_SCRIPT = """
const DATA = JSON.parse(document.getElementById('pov-data').textContent);
const list = document.getElementById('samples');
const search = document.getElementById('search');
const condFilter = document.getElementById('condition-filter');
const sortBy = document.getElementById('sort-by');

const conditions = [...new Set(DATA.samples.map(s => s.condition).filter(Boolean))].sort();
for (const c of conditions) {
  const opt = document.createElement('option');
  opt.value = c; opt.textContent = c;
  condFilter.appendChild(opt);
}
for (const label of DATA.metricLabels) {
  const opt = document.createElement('option');
  opt.value = label; opt.textContent = 'score: ' + label;
  sortBy.appendChild(opt);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

function mediaHtml(sample) {
  if (!sample.mediaSrc) return '';
  if (sample.mediaKind === 'video') {
    return `<video controls preload="none" src="${escapeHtml(sample.mediaSrc)}"></video>`;
  }
  if (sample.mediaKind === 'image') {
    return `<img loading="lazy" src="${escapeHtml(sample.mediaSrc)}" alt="${escapeHtml(sample.id)}">`;
  }
  return `<a href="${escapeHtml(sample.mediaSrc)}">open media</a>`;
}

function field(label, value) {
  const body = value ? escapeHtml(value) : '<span class="missing">(empty)</span>';
  return `<div class="field"><div class="field-label">${escapeHtml(label)}</div>
          <div class="field-value">${body}</div></div>`;
}

function cardHtml(sample) {
  const chips = Object.entries(sample.scores)
    .map(([k, v]) => `${k}=${v.toFixed(3)}`).join('  ');
  const details = Object.entries(sample.details)
    .map(([k, v]) => `<span>${escapeHtml(k)}: ${escapeHtml(v)}</span>`).join('');
  const media = mediaHtml(sample);
  return `<details class="card">
    <summary>
      <span class="sid">${escapeHtml(sample.id)}</span>
      <span class="tag">${escapeHtml(sample.condition)}</span>
      <span class="score-chip">${escapeHtml(chips)}</span>
    </summary>
    <div class="body${media ? ' has-media' : ''}">
      ${media ? `<div>${media}</div>` : ''}
      <div>
        ${field('ground truth', sample.groundTruth)}
        ${field('model output', sample.modelOutput)}
        <div class="kv">${details}</div>
      </div>
    </div>
  </details>`;
}

function render() {
  const query = search.value.trim().toLowerCase();
  const condition = condFilter.value;
  const sortKey = sortBy.value;

  let rows = DATA.samples.filter(s => {
    if (condition && s.condition !== condition) return false;
    if (!query) return true;
    return (s.id + ' ' + s.groundTruth + ' ' + s.modelOutput).toLowerCase().includes(query);
  });

  if (sortKey) {
    rows = [...rows].sort((a, b) => (a.scores[sortKey] ?? -Infinity) - (b.scores[sortKey] ?? -Infinity));
  }

  document.getElementById('shown').textContent = rows.length;
  list.innerHTML = rows.length
    ? rows.map(cardHtml).join('')
    : '<p class="empty">No samples match this filter.</p>';
}

search.addEventListener('input', render);
condFilter.addEventListener('change', render);
sortBy.addEventListener('change', render);
render();
"""

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <p class="sub">Generated by pov. Videos are linked relative to this file — keep it
  beside the run directory.</p>

  <div class="stats">{header}</div>

  <h2>Scores by condition</h2>
  {summary_table}

  <h2>Samples (<span id="shown">0</span>)</h2>
  <div class="controls">
    <input type="search" id="search" placeholder="Search id, ground truth, or model output…">
    <select id="condition-filter"><option value="">All conditions</option></select>
    <select id="sort-by"><option value="">Default order</option></select>
  </div>
  <div id="samples"></div>

  <footer>pov report · scores shown are per-sample values from scored.csv</footer>
</div>
<script type="application/json" id="pov-data">{data}</script>
<script>{script}</script>
</body>
</html>
"""
