#!/usr/bin/env bash
#
# Download the How2Sign source data used by the `asl` experiment.
#
#   ./scripts/download_how2sign.sh              # val split (default, ~1.7 GB)
#   ./scripts/download_how2sign.sh --split test # test split (~2.2 GB)
#   ./scripts/download_how2sign.sh --split train --yes   # train split (~31 GB!)
#   ./scripts/download_how2sign.sh --dest /Volumes/big-disk/how2sign
#
# Source: https://how2sign.github.io/  (Duarte et al., CVPR 2021)
# Licence: CC BY-NC 4.0 — research use only, non-commercial, not redistributable.
#
# Downloads two things per split:
#   1. Green Screen RGB *Clips*, frontal view — one mp4 per sentence.
#      These are the sentence-level clips the pipeline expects (NOT the much
#      larger full-length "Green Screen RGB Videos").
#   2. The English translation CSV (original alignment), which supplies the
#      SENTENCE ground truth and the START/END timestamps.
#
# Files land in <dest>/ and are gitignored. Nothing outside <dest> is touched.

set -euo pipefail

SPLIT="val"
DEST=""
ASSUME_YES=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="${2:-}"; shift 2 ;;
        --dest)  DEST="${2:-}";  shift 2 ;;
        --yes|-y) ASSUME_YES=1;  shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

DEST="${DEST:-$REPO_ROOT/data/asl_source}"

# ── Google Drive file ids, from https://how2sign.github.io/#download ──────────
# Green Screen RGB Clips (Frontal) — sentence-level mp4s
case "$SPLIT" in
    val)   CLIPS_ID="1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO"; CLIPS_SIZE="1.7 GB" ;;
    test)  CLIPS_ID="1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG"; CLIPS_SIZE="2.2 GB" ;;
    train) CLIPS_ID="1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW"; CLIPS_SIZE="31 GB"  ;;
    *) echo "error: --split must be val, test, or train (got '$SPLIT')" >&2; exit 1 ;;
esac

# English Translation (Original alignment) — has VIDEO_ID/SENTENCE_NAME/START/END/SENTENCE
case "$SPLIT" in
    val)   CSV_ID="1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3" ;;
    test)  CSV_ID="1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG" ;;
    train) CSV_ID="1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f" ;;
esac

CSV_PATH="$DEST/how2sign_${SPLIT}.csv"
VIDEO_DIR="$DEST/${SPLIT}_raw_videos"

echo "How2Sign downloader"
echo "  split      : $SPLIT"
echo "  clips      : ~$CLIPS_SIZE (Green Screen RGB Clips, frontal view)"
echo "  destination: $DEST"
echo
echo "  Licence: CC BY-NC 4.0 — research use only. By downloading you accept"
echo "  the terms at https://how2sign.github.io/"
echo

if [[ "$SPLIT" == "train" && "$ASSUME_YES" -eq 0 ]]; then
    echo "The train split is ~31 GB. Re-run with --yes to confirm." >&2
    exit 1
fi

# ── Dependencies ─────────────────────────────────────────────────────────────
# Google Drive interposes a virus-scan warning for large files; gdown handles
# the confirmation token. Plain curl/wget will silently save an HTML page.
if ! command -v gdown >/dev/null 2>&1; then
    echo "error: 'gdown' is required to download from Google Drive." >&2
    echo "       install it with:  pip install gdown" >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "error: 'unzip' is required." >&2
    exit 1
fi

mkdir -p "$DEST"

# ── 1. Translation CSV ───────────────────────────────────────────────────────
if [[ -s "$CSV_PATH" ]]; then
    echo "[1/2] translation CSV already present: $CSV_PATH"
else
    echo "[1/2] downloading translation CSV…"
    tmp="$DEST/.how2sign_${SPLIT}_csv.download"
    rm -rf "$tmp"; mkdir -p "$tmp"
    gdown "https://drive.google.com/uc?id=$CSV_ID" -O "$tmp/download" --quiet

    # The link may serve the raw CSV or a zip containing it.
    if unzip -tq "$tmp/download" >/dev/null 2>&1; then
        unzip -joq "$tmp/download" -d "$tmp/extracted"
        # -print -quit rather than `| head -1`: under `set -o pipefail` the
        # closed pipe kills find with SIGPIPE and aborts the whole script.
        found="$(find "$tmp/extracted" -name '*.csv' -print -quit)"
        [[ -n "$found" ]] || { echo "error: no CSV inside the archive" >&2; exit 1; }
        mv "$found" "$CSV_PATH"
    else
        mv "$tmp/download" "$CSV_PATH"
    fi
    rm -rf "$tmp"
    echo "      saved $CSV_PATH"
fi

# Sanity-check the header so a Drive HTML error page cannot pass as data.
if ! head -1 "$CSV_PATH" | grep -q "SENTENCE_NAME"; then
    echo "error: $CSV_PATH does not look like a How2Sign translation file." >&2
    echo "       (no SENTENCE_NAME column — the download may have failed)" >&2
    echo "       Delete it and re-run, or download manually from" >&2
    echo "       https://how2sign.github.io/" >&2
    exit 1
fi

# ── 2. Sentence-level clips ──────────────────────────────────────────────────
existing=0
if [[ -d "$VIDEO_DIR" ]]; then
    existing="$(find "$VIDEO_DIR" -name '*.mp4' | wc -l | tr -d ' ')"
fi

if [[ "$existing" -gt 0 ]]; then
    echo "[2/2] clips already present: $VIDEO_DIR ($existing mp4 files)"
else
    echo "[2/2] downloading clips (~$CLIPS_SIZE) — this takes a while…"
    zip_path="$DEST/${SPLIT}_rgb_front_clips.zip"
    if [[ ! -s "$zip_path" ]]; then
        gdown "https://drive.google.com/uc?id=$CLIPS_ID" -O "$zip_path"
    fi

    echo "      extracting…"
    staging="$DEST/.extract_${SPLIT}"
    rm -rf "$staging"; mkdir -p "$staging"
    unzip -qo "$zip_path" -d "$staging"

    # The archive's internal layout has changed across releases, so locate the
    # directory that actually holds the mp4s instead of assuming a path.
    # `find … | head -1` would abort here: head closes the pipe after the first
    # of ~1,700 paths, find dies with SIGPIPE, and `set -o pipefail` propagates
    # exit 141 — the whole download silently fails right after extracting.
    sample="$(find "$staging" -name '*.mp4' -print -quit)"
    if [[ -z "$sample" ]]; then
        echo "error: no .mp4 files found inside $zip_path" >&2
        exit 1
    fi
    mkdir -p "$VIDEO_DIR"
    find "$(dirname "$sample")" -name '*.mp4' -exec mv -n {} "$VIDEO_DIR"/ \;
    rm -rf "$staging"

    count="$(find "$VIDEO_DIR" -name '*.mp4' | wc -l | tr -d ' ')"
    echo "      extracted $count clips to $VIDEO_DIR"
    echo "      (the zip is kept at $zip_path — delete it to reclaim space)"
fi

# ── Report ───────────────────────────────────────────────────────────────────
csv_rows="$(( $(wc -l < "$CSV_PATH") - 1 ))"
clip_count="$(find "$VIDEO_DIR" -name '*.mp4' | wc -l | tr -d ' ')"

echo
echo "Done."
echo "  translation rows : $csv_rows"
echo "  clips on disk    : $clip_count"
echo
echo "Point the ASL config at these paths:"
echo
echo "  pov generate -c configs/asl.yaml \\"
echo "    --set params.metadata_csv=$CSV_PATH \\"
echo "    --set params.video_dir=$VIDEO_DIR"
echo
echo "…or edit configs/asl.yaml directly:"
echo
echo "  params:"
echo "    metadata_csv: $CSV_PATH"
echo "    video_dir: $VIDEO_DIR"
