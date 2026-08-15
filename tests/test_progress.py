"""Progress reporting."""

from __future__ import annotations

import io
import threading

import pytest

from pov.progress import Progress


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


class TestProgress:
    def test_counts_updates(self):
        with Progress(total=10, disable=True) as bar:
            bar.update()
            bar.update(3)
        assert bar.count == 4

    def test_renders_to_a_terminal(self):
        stream = FakeTTY()
        with Progress(total=4, desc="chess", unit="clip", stream=stream) as bar:
            bar.update(4)
        assert "chess" in stream.getvalue()

    def test_stays_silent_on_a_non_terminal(self):
        # A redirected log must not collect thousands of carriage returns.
        stream = io.StringIO()
        with Progress(total=3, desc="chess", stream=stream) as bar:
            bar.update(3)
        assert stream.getvalue() == ""

    def test_disable_true_writes_nothing_even_to_a_terminal(self):
        stream = FakeTTY()
        with Progress(total=3, desc="chess", stream=stream, disable=True) as bar:
            bar.update(3)
        assert stream.getvalue() == ""
        assert bar.count == 3

    def test_non_terminal_reports_periodically_on_a_long_run(self):
        stream = io.StringIO()
        bar = Progress(total=100, desc="asl", unit="clip", stream=stream)
        bar.FALLBACK_INTERVAL = 0.0  # report on the next update
        bar.update(50)
        bar.close()
        output = stream.getvalue()
        assert "asl" in output and "50/100" in output

    def test_updates_are_thread_safe(self):
        # Workers finish out of order and call update concurrently.
        bar = Progress(total=400, disable=True)
        threads = [
            threading.Thread(target=lambda: [bar.update() for _ in range(100)])
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        bar.close()
        assert bar.count == 400

    def test_unknown_total_is_allowed(self):
        with Progress(total=None, disable=True) as bar:
            bar.update()
        assert bar.count == 1

    def test_set_description_does_not_raise_when_disabled(self):
        with Progress(total=1, disable=True) as bar:
            bar.set_description("measuring durations")
        assert bar.desc == "measuring durations"

    def test_close_is_idempotent(self):
        bar = Progress(total=1, stream=FakeTTY())
        bar.close()
        bar.close()

    def test_falls_back_when_tqdm_is_unavailable_on_a_terminal(self, monkeypatch):
        # Without tqdm we must still report, not go silent on a terminal.
        import pov.progress as module

        monkeypatch.setattr(module, "_tqdm", None)
        stream = FakeTTY()
        bar = Progress(total=10, desc="chess", unit="clip", stream=stream)
        assert bar._bar is None and bar._fallback is True
        bar.FALLBACK_INTERVAL = 0.0
        bar.update(5)
        bar.close()
        assert "5/10" in stream.getvalue()


class TestGeneratorWiring:
    def test_quiet_disables_the_bar(self, chess_config):
        from pov.config import Config
        from pov.registry import build_generator

        generator = build_generator(Config.from_mapping(chess_config))
        generator.show_progress = False
        bar = generator.progress(10)
        assert bar._bar is None
        bar.update()
        bar.close()

    def test_default_leaves_the_decision_to_the_stream(self, chess_config):
        # Must not force the bar on: under pytest, or with output redirected,
        # stderr is not a terminal and the bar should stay quiet.
        from pov.config import Config
        from pov.registry import build_generator

        generator = build_generator(Config.from_mapping(chess_config))
        assert generator.show_progress is True
        bar = generator.progress(10)
        assert bar._bar is None  # stderr is captured by pytest
        bar.close()
