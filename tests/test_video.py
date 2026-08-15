"""Video encoding: pure-python parts always, real encoding when ffmpeg exists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pov.config import VideoConfig
from pov.video import (
    EncodeSettings,
    VideoError,
    VideoStats,
    VideoWriter,
    as_rgb_array,
    cut_clip,
    probe_video,
    reduce_holds,
    stream_codec_name,
    write_timeline,
)
from tests.conftest import requires_ffmpeg


def frame(value: int = 128, width: int = 64, height: int = 48) -> np.ndarray:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = value
    return array


# ── Pure logic ────────────────────────────────────────────────────────────────


class TestReduceHolds:
    @pytest.mark.parametrize(
        "counts,expected",
        [
            ([15, 15, 15], 15),
            ([60, 4, 60, 4], 4),
            ([7, 3], 1),
            ([10], 10),
            ([], 1),
            ([0, 0], 1),
            ([100, 50, 25], 25),
        ],
    )
    def test_gcd(self, counts, expected):
        assert reduce_holds(counts) == expected

    def test_rejects_negative(self):
        with pytest.raises(VideoError, match=">= 0"):
            reduce_holds([5, -1])


class TestAsRgbArray:
    def test_from_pil(self):
        array = as_rgb_array(Image.new("RGB", (8, 4), (10, 20, 30)))
        assert array.shape == (4, 8, 3) and array.dtype == np.uint8
        assert tuple(array[0, 0]) == (10, 20, 30)

    def test_greyscale_expands_to_three_channels(self):
        array = as_rgb_array(np.zeros((4, 4), dtype=np.uint8))
        assert array.shape == (4, 4, 3)

    def test_rgba_drops_alpha(self):
        array = as_rgb_array(np.zeros((4, 4, 4), dtype=np.uint8))
        assert array.shape == (4, 4, 3)

    def test_float_is_clipped_and_cast(self):
        array = as_rgb_array(np.full((2, 2, 3), 300.0))
        assert array.dtype == np.uint8 and array.max() == 255

    def test_rejects_wrong_rank(self):
        with pytest.raises(VideoError, match="2-D or 3-D"):
            as_rgb_array(np.zeros((2, 2, 2, 2)))

    def test_rejects_two_channels(self):
        with pytest.raises(VideoError, match="3 channels"):
            as_rgb_array(np.zeros((4, 4, 2), dtype=np.uint8))

    def test_result_is_contiguous(self):
        array = as_rgb_array(np.zeros((6, 6, 3), dtype=np.uint8)[::2])
        assert array.flags["C_CONTIGUOUS"]


class TestEncodeSettings:
    def test_from_config(self):
        settings = EncodeSettings.from_config(VideoConfig(fps=24, crf=20, preset="fast"))
        assert (settings.fps, settings.crf, settings.preset) == (24, 20, "fast")

    def test_defaults(self):
        settings = EncodeSettings()
        assert settings.codec == "libx264" and settings.pix_fmt == "yuv420p"


class TestVideoStats:
    def test_pipe_savings(self):
        stats = VideoStats(Path("x.mp4"), 2, 2, 30.0, 100, 3.3, "libx264", piped_frames=10)
        assert stats.pipe_savings == pytest.approx(0.9)

    def test_pipe_savings_with_no_frames(self):
        assert VideoStats(Path("x"), 2, 2, 30.0, 0, 0.0, "libx264").pipe_savings == 0.0


class TestWriterValidation:
    def test_rejects_odd_dimensions(self, tmp_path):
        with pytest.raises(VideoError, match="even dimensions"):
            VideoWriter(tmp_path / "v.mp4", (65, 48), EncodeSettings())

    def test_rejects_zero_size(self, tmp_path):
        with pytest.raises(VideoError, match="invalid frame size"):
            VideoWriter(tmp_path / "v.mp4", (0, 48), EncodeSettings())

    def test_empty_timeline_rejected(self, tmp_path):
        with pytest.raises(VideoError, match="no frames"):
            write_timeline(tmp_path / "v.mp4", [], EncodeSettings())

    def test_zero_count_segments_are_dropped(self, tmp_path):
        with pytest.raises(VideoError, match="no frames"):
            write_timeline(tmp_path / "v.mp4", [(frame(), 0)], EncodeSettings())

    def test_probe_missing_file(self, tmp_path):
        with pytest.raises(VideoError, match="not found"):
            probe_video(tmp_path / "nope.mp4")

    def test_cut_missing_source(self, tmp_path):
        with pytest.raises(VideoError, match="not found"):
            cut_clip(tmp_path / "nope.mp4", tmp_path / "out.mp4")


# ── Real encoding ─────────────────────────────────────────────────────────────


@requires_ffmpeg
@pytest.mark.integration
class TestEncoding:
    def test_uniform_holds_produce_exact_frame_count(self, tmp_path):
        path = tmp_path / "uniform.mp4"
        segments = [(frame(i * 20), 15) for i in range(8)]
        stats = write_timeline(path, segments, EncodeSettings(fps=30, preset="ultrafast"))

        assert stats.n_frames == 120
        assert stats.piped_frames == 8          # GCD 15 -> one pipe write per frame
        assert probe_video(path).n_frames == 120

    def test_mixed_holds_produce_exact_frame_count(self, tmp_path):
        path = tmp_path / "mixed.mp4"
        segments = []
        for i in range(5):
            segments.append((frame(i * 40), 60))
            segments.append((frame(255), 4))
        stats = write_timeline(path, segments, EncodeSettings(fps=30, preset="ultrafast"))

        assert stats.n_frames == 320
        assert stats.piped_frames == 80         # GCD 4
        assert probe_video(path).n_frames == 320

    def test_coprime_holds_still_exact(self, tmp_path):
        path = tmp_path / "coprime.mp4"
        stats = write_timeline(
            path, [(frame(10), 7), (frame(200), 3)],
            EncodeSettings(fps=30, preset="ultrafast"),
        )
        assert stats.n_frames == 10 and stats.piped_frames == 10
        assert probe_video(path).n_frames == 10

    def test_single_frame(self, tmp_path):
        path = tmp_path / "one.mp4"
        stats = write_timeline(path, [(frame(), 1)], EncodeSettings(fps=30, preset="ultrafast"))
        assert stats.n_frames == 1
        assert probe_video(path).n_frames == 1

    def test_duration_matches_frames_over_fps(self, tmp_path):
        path = tmp_path / "dur.mp4"
        stats = write_timeline(
            path, [(frame(), 60)], EncodeSettings(fps=30, preset="ultrafast")
        )
        assert stats.duration_sec == pytest.approx(2.0)
        assert probe_video(path).duration_sec == pytest.approx(2.0, abs=0.1)

    def test_reported_size_matches_frames(self, tmp_path):
        path = tmp_path / "size.mp4"
        stats = write_timeline(
            path, [(frame(width=32, height=16), 4)],
            EncodeSettings(fps=30, preset="ultrafast"),
        )
        assert (stats.width, stats.height) == (32, 16)
        probed = probe_video(path)
        assert (probed.width, probed.height) == (32, 16)

    def test_pil_frames_accepted(self, tmp_path):
        path = tmp_path / "pil.mp4"
        image = Image.new("RGB", (32, 16), (200, 10, 10))
        stats = write_timeline(path, [(image, 3)], EncodeSettings(fps=30, preset="ultrafast"))
        assert stats.n_frames == 3

    def test_file_size_is_recorded(self, tmp_path):
        path = tmp_path / "sz.mp4"
        stats = write_timeline(path, [(frame(), 10)], EncodeSettings(fps=30, preset="ultrafast"))
        assert stats.file_size_bytes > 0
        assert stats.file_size_bytes == path.stat().st_size

    def test_writer_rejects_mismatched_frame_size(self, tmp_path):
        writer = VideoWriter(tmp_path / "v.mp4", (64, 48), EncodeSettings(preset="ultrafast"))
        with writer:
            with pytest.raises(VideoError, match="writer expects"):
                writer.write(frame(width=32, height=16))
            writer.write(frame())

    def test_writer_with_no_frames_raises_and_leaves_no_file(self, tmp_path):
        path = tmp_path / "empty.mp4"
        writer = VideoWriter(path, (64, 48), EncodeSettings(preset="ultrafast"))
        writer.open()
        with pytest.raises(VideoError, match="no frames"):
            writer.close()
        assert not path.exists()

    def test_abort_removes_partial_output(self, tmp_path):
        path = tmp_path / "aborted.mp4"
        writer = VideoWriter(path, (64, 48), EncodeSettings(preset="ultrafast"))
        with writer:
            writer.write(frame(), repeat=3)
            writer.abort()
        assert not path.exists()

    def test_exception_inside_context_aborts(self, tmp_path):
        path = tmp_path / "boom.mp4"
        with pytest.raises(RuntimeError):
            with VideoWriter(path, (64, 48), EncodeSettings(preset="ultrafast")) as writer:
                writer.write(frame())
                raise RuntimeError("boom")
        assert not path.exists()

    def test_negative_repeat_rejected(self, tmp_path):
        with VideoWriter(tmp_path / "v.mp4", (64, 48), EncodeSettings(preset="ultrafast")) as w:
            with pytest.raises(VideoError, match="repeat"):
                w.write(frame(), repeat=-1)
            w.write(frame())

    def test_zero_repeat_writes_nothing(self, tmp_path):
        with VideoWriter(tmp_path / "v.mp4", (64, 48), EncodeSettings(preset="ultrafast")) as w:
            w.write(frame(), repeat=0)
            assert w.frames_written == 0
            w.write(frame())

    def test_cut_clip_trims_and_rescales(self, tmp_path):
        source = tmp_path / "src.mp4"
        write_timeline(
            source, [(frame(i * 10, width=64, height=48), 30) for i in range(4)],
            EncodeSettings(fps=30, preset="ultrafast"),
        )
        dest = tmp_path / "cut.mp4"
        stats = cut_clip(
            source, dest, EncodeSettings(fps=30, preset="ultrafast"),
            start=1.0, duration=1.0, scale_height=24,
        )
        assert stats.height == 24
        assert stats.width % 2 == 0
        assert stats.duration_sec == pytest.approx(1.0, abs=0.2)

    def test_cut_clip_rejects_bad_ranges(self, tmp_path):
        source = tmp_path / "src.mp4"
        write_timeline(source, [(frame(), 10)], EncodeSettings(fps=30, preset="ultrafast"))
        with pytest.raises(VideoError, match="start"):
            cut_clip(source, tmp_path / "a.mp4", start=-1.0)
        with pytest.raises(VideoError, match="duration"):
            cut_clip(source, tmp_path / "b.mp4", duration=0)

    def test_probe_reports_codec_and_fps(self, tmp_path):
        path = tmp_path / "probe.mp4"
        write_timeline(path, [(frame(), 30)], EncodeSettings(fps=30, preset="ultrafast"))
        stats = probe_video(path)
        assert stats.codec == "h264"
        assert stats.fps == pytest.approx(30.0, abs=0.5)

    def test_written_codec_matches_the_probed_codec(self, tmp_path):
        """A resumed row probes the file; a fresh row does not. They must agree.

        `write_timeline` used to report the *encoder* (`libx264`) while
        `probe_video` and `cut_clip` report the *stream* codec (`h264`), so the
        same clip changed its manifest `codec` depending on whether the run had
        been resumed.
        """
        path = tmp_path / "codec.mp4"
        written = write_timeline(
            path, [(frame(), 10)], EncodeSettings(fps=30, preset="ultrafast")
        )
        assert written.codec == probe_video(path).codec == "h264"

    def test_cut_clip_and_write_timeline_agree_on_codec(self, tmp_path):
        settings = EncodeSettings(fps=30, preset="ultrafast")
        source = tmp_path / "src.mp4"
        written = write_timeline(source, [(frame(), 30)], settings)
        cut = cut_clip(source, tmp_path / "cut.mp4", settings, duration=0.5)
        assert written.codec == cut.codec


class TestStreamCodecName:
    """The encoder → probed-codec mapping behind the manifest's `codec` column."""

    def test_maps_known_encoders(self):
        assert stream_codec_name("libx264") == "h264"
        assert stream_codec_name("libx265") == "hevc"
        assert stream_codec_name("libvpx-vp9") == "vp9"

    def test_passes_through_unknown_and_self_named_encoders(self):
        assert stream_codec_name("mpeg4") == "mpeg4"
        assert stream_codec_name("some_future_encoder") == "some_future_encoder"

    def test_default_encoder_is_mapped(self):
        """The default must be mapped, or every default run mislabels its rows."""
        assert stream_codec_name(EncodeSettings().codec) == "h264"
