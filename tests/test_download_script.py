"""End-to-end checks for scripts/download_how2sign.sh.

The script runs under `set -euo pipefail`, which makes any pipeline whose
producer is killed by SIGPIPE abort the whole run. `find … | head -1` over the
~1,700 extracted clips does exactly that, so the download used to die silently
(exit 141) immediately after unpacking the 1.7 GB archive, leaving no clips
directory behind. These tests drive the extraction path with a prepared archive
so that regression cannot come back.

Nothing here touches the network: the translation CSV and the clips zip are
pre-placed in the destination, so the script's own "already downloaded" checks
short-circuit both `gdown` calls. `gdown` is stubbed with a script that fails
loudly if it is ever invoked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_how2sign.sh"

requires_shell_tools = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("unzip")),
    reason="bash/unzip not on PATH",
)

CSV_HEADER = "VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART\tEND\tSENTENCE\n"

# The pipe buffer is 64 KiB; the producer only takes SIGPIPE if it is still
# writing when the consumer exits. Enough long paths to blow past that makes
# the failure deterministic rather than a coin flip.
N_CLIPS = 1200


def _make_dest(tmp_path: Path, n_clips: int = N_CLIPS, inner_dir: str = "raw_videos") -> Path:
    dest = tmp_path / "asl_source"
    dest.mkdir()

    (dest / "how2sign_val.csv").write_text(
        CSV_HEADER + "vid\tvid-1-rgb_front\tvid_0\tvid_0-1-rgb_front\t0.0\t1.0\thello\n",
        encoding="utf-8",
    )

    zip_path = dest / "val_rgb_front_clips.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for index in range(n_clips):
            archive.writestr(f"{inner_dir}/a_long_enough_clip_name_{index:06d}-rgb_front.mp4", b"x")
    return dest


def _run(dest: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the script with a `gdown` stub that fails if it is ever called."""
    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir(exist_ok=True)
    gdown = stub_dir / "gdown"
    gdown.write_text("#!/bin/sh\necho 'gdown must not be called' >&2\nexit 99\n")
    gdown.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(SCRIPT), "--dest", str(dest), *args],
        capture_output=True,
        text=True,
        env=env,
    )


@requires_shell_tools
@pytest.mark.slow
def test_extraction_completes_with_many_clips(tmp_path: Path) -> None:
    """The regression: `find | head` used to kill the script with SIGPIPE (141)."""
    dest = _make_dest(tmp_path)

    result = _run(dest, tmp_path)

    assert result.returncode == 0, (
        f"exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode != 141, "script died on a SIGPIPE'd pipeline"
    video_dir = dest / "val_raw_videos"
    assert video_dir.is_dir(), "clips directory was never created"
    assert len(list(video_dir.glob("*.mp4"))) == N_CLIPS
    assert "Done." in result.stdout


@requires_shell_tools
@pytest.mark.slow
def test_rerun_is_a_no_op(tmp_path: Path) -> None:
    """A second run sees the clips already on disk and re-extracts nothing."""
    dest = _make_dest(tmp_path, n_clips=20)
    assert _run(dest, tmp_path).returncode == 0

    result = _run(dest, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "already present" in result.stdout


@requires_shell_tools
def test_archive_without_clips_is_reported(tmp_path: Path) -> None:
    """An archive with no mp4s must fail with a message, not a bare exit code."""
    dest = _make_dest(tmp_path, n_clips=0)
    with zipfile.ZipFile(dest / "val_rgb_front_clips.zip", "w") as archive:
        archive.writestr("readme.txt", b"not a clip")

    result = _run(dest, tmp_path)

    assert result.returncode == 1
    assert "no .mp4 files found" in result.stderr


@requires_shell_tools
def test_bad_split_is_rejected(tmp_path: Path) -> None:
    result = _run(tmp_path / "unused", tmp_path, "--split", "bogus")

    assert result.returncode == 1
    assert "must be val, test, or train" in result.stderr


@requires_shell_tools
def test_csv_that_is_not_how2sign_is_rejected(tmp_path: Path) -> None:
    """A Drive error page saved as the CSV must not pass as data."""
    dest = _make_dest(tmp_path, n_clips=1)
    (dest / "how2sign_val.csv").write_text("<html>Quota exceeded</html>\n", encoding="utf-8")

    result = _run(dest, tmp_path)

    assert result.returncode == 1
    assert "does not look like a How2Sign translation file" in result.stderr
