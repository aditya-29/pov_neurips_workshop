"""YAML configuration loading, validation, and resolution.

A config file looks like::

    experiment: chess
    run:
      output_root: data
      seed: 1234
    video:
      fps: 30
    params:
      num_games: 20
      durations: [{label: 5s, seconds: 5}]

`params` is experiment-specific and validated by the experiment itself; the
rest is common to every experiment and validated here.

Design notes
------------
* Unknown keys are an **error**, not a warning. A typo like ``fsp: 30`` must
  never silently fall back to the default — that would invalidate a whole
  generation run.
* The resolved config is snapshotted next to the generated data and hashed, so
  every row of the manifest can point back at exactly what produced it.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared
    raise ImportError("pov requires PyYAML: pip install pyyaml") from exc


class ConfigError(ValueError):
    """Raised for any malformed, missing, or unknown configuration value."""


_MISSING = object()

# Run ids are used as directory names; keep them filesystem-safe.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ──────────────────────────────────────────────────────────────────────────────
# Mapping reader — gives precise, path-qualified error messages
# ──────────────────────────────────────────────────────────────────────────────


class Reader:
    """Validating accessor over a mapping.

    Tracks which keys were consumed so :meth:`done` can reject unknown ones.

        r = Reader({"fps": 30}, path="video")
        fps = r.int("fps", default=30, min=1)
        r.done()
    """

    def __init__(self, mapping: Any, path: str = ""):
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, Mapping):
            raise ConfigError(
                f"{path or 'config'}: expected a mapping, got {_typename(mapping)}"
            )
        self._data: dict = dict(mapping)
        self._path = path
        self._seen: set[str] = set()

    # -- internals ---------------------------------------------------------

    def _where(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _get(self, key: str, default: Any) -> Any:
        self._seen.add(key)
        if key in self._data:
            value = self._data[key]
            if value is None and default is not _MISSING:
                return default
            return value
        if default is _MISSING:
            raise ConfigError(f"{self._where(key)}: required key is missing")
        return default

    # -- introspection -----------------------------------------------------

    def keys(self) -> list[str]:
        """Keys present in the mapping.

        For sections whose keys are user-chosen (e.g. speed names), where the
        schema cannot enumerate them ahead of time.
        """
        return list(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    # -- typed accessors ---------------------------------------------------

    def raw(self, key: str, default: Any = _MISSING) -> Any:
        return self._get(key, default)

    def str(self, key: str, default: Any = _MISSING, choices: Sequence[str] | None = None) -> str:
        value = self._get(key, default)
        if not isinstance(value, str):
            raise ConfigError(
                f"{self._where(key)}: expected a string, got {_typename(value)}"
            )
        if choices is not None and value not in choices:
            raise ConfigError(
                f"{self._where(key)}: {value!r} is not one of {sorted(choices)}"
            )
        return value

    def int(
        self,
        key: str,
        default: Any = _MISSING,
        min: int | None = None,
        max: int | None = None,
    ) -> int:
        value = self._get(key, default)
        # bool is a subclass of int; reject it so `fps: true` is not silently 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"{self._where(key)}: expected an integer, got {_typename(value)}"
            )
        self._check_range(key, value, min, max)
        return value

    def float(
        self,
        key: str,
        default: Any = _MISSING,
        min: float | None = None,
        max: float | None = None,
    ) -> float:
        value = self._get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{self._where(key)}: expected a number, got {_typename(value)}"
            )
        value = float(value)
        self._check_range(key, value, min, max)
        return value

    def bool(self, key: str, default: Any = _MISSING) -> bool:
        value = self._get(key, default)
        if not isinstance(value, bool):
            raise ConfigError(
                f"{self._where(key)}: expected true or false, got {_typename(value)}"
            )
        return value

    def path(self, key: str, default: Any = _MISSING) -> Path:
        value = self._get(key, default)
        if isinstance(value, Path):
            return value
        if not isinstance(value, str):
            raise ConfigError(
                f"{self._where(key)}: expected a path string, got {_typename(value)}"
            )
        if not value:
            raise ConfigError(f"{self._where(key)}: path must not be empty")
        return Path(value).expanduser()

    def optional_str(self, key: str, default: Any = None) -> str | None:
        self._seen.add(key)
        value = self._data.get(key, default)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(
                f"{self._where(key)}: expected a string or null, got {_typename(value)}"
            )
        return value

    def list(
        self,
        key: str,
        default: Any = _MISSING,
        item_type: type | tuple[type, ...] | None = None,
        min_len: int = 0,
    ) -> list:
        value = self._get(key, default)
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ConfigError(
                f"{self._where(key)}: expected a list, got {_typename(value)}"
            )
        items = list(value)
        if len(items) < min_len:
            raise ConfigError(
                f"{self._where(key)}: needs at least {min_len} item(s), got {len(items)}"
            )
        if item_type is not None:
            for i, item in enumerate(items):
                if isinstance(item, bool) and item_type is not bool:
                    raise ConfigError(
                        f"{self._where(key)}[{i}]: expected {_typename_of(item_type)}, got bool"
                    )
                if not isinstance(item, item_type):
                    raise ConfigError(
                        f"{self._where(key)}[{i}]: expected {_typename_of(item_type)}, "
                        f"got {_typename(item)}"
                    )
        return items

    def reader(self, key: str, default: Any = None) -> "Reader":
        """Return a nested Reader for a sub-mapping."""
        value = self._get(key, default if default is not None else {})
        return Reader(value, path=self._where(key))

    def _check_range(self, key: str, value: float, min: float | None, max: float | None) -> None:
        if min is not None and value < min:
            raise ConfigError(f"{self._where(key)}: must be >= {min}, got {value}")
        if max is not None and value > max:
            raise ConfigError(f"{self._where(key)}: must be <= {max}, got {value}")

    # -- completion --------------------------------------------------------

    def done(self) -> None:
        """Raise if any key in the mapping was never read."""
        unknown = sorted(set(self._data) - self._seen)
        if unknown:
            known = sorted(self._seen)
            where = self._path or "config"
            raise ConfigError(
                f"{where}: unknown key(s) {unknown}. Known keys here: {known}"
            )


def _typename(value: Any) -> str:
    return type(value).__name__


def _typename_of(t: type | tuple[type, ...]) -> str:
    if isinstance(t, tuple):
        return " or ".join(x.__name__ for x in t)
    return t.__name__


# ──────────────────────────────────────────────────────────────────────────────
# Common config sections
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VideoConfig:
    """Encoder settings shared by every experiment."""

    fps: int = 30
    crf: int = 18
    preset: str = "veryfast"
    tune: str | None = "stillimage"
    pix_fmt: str = "yuv420p"
    codec: str = "libx264"
    threads: int = 0  # 0 = let ffmpeg decide

    PRESETS = (
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    )
    TUNES = ("film", "animation", "grain", "stillimage", "fastdecode", "zerolatency")

    @classmethod
    def from_reader(cls, r: Reader) -> "VideoConfig":
        cfg = cls(
            fps=r.int("fps", 30, min=1, max=240),
            crf=r.int("crf", 18, min=0, max=51),
            preset=r.str("preset", "veryfast", choices=cls.PRESETS),
            tune=r.optional_str("tune", "stillimage"),
            pix_fmt=r.str("pix_fmt", "yuv420p"),
            codec=r.str("codec", "libx264"),
            threads=r.int("threads", 0, min=0, max=64),
        )
        r.done()
        if cfg.tune is not None and cfg.tune not in cls.TUNES:
            raise ConfigError(
                f"video.tune: {cfg.tune!r} is not one of {sorted(cls.TUNES)} (or null)"
            )
        return cfg

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "crf": self.crf,
            "preset": self.preset,
            "tune": self.tune,
            "pix_fmt": self.pix_fmt,
            "codec": self.codec,
            "threads": self.threads,
        }


@dataclass(frozen=True)
class RunConfig:
    """Where output goes and how the run is executed."""

    output_root: Path = Path("data")
    run_id: str | None = None
    seed: int = 1234
    workers: int = 8
    overwrite: bool = False
    resume: bool = True

    @classmethod
    def from_reader(cls, r: Reader) -> "RunConfig":
        cfg = cls(
            output_root=r.path("output_root", Path("data")),
            run_id=r.optional_str("run_id", None),
            seed=r.int("seed", 1234, min=0),
            workers=r.int("workers", 8, min=1, max=256),
            overwrite=r.bool("overwrite", False),
            resume=r.bool("resume", True),
        )
        r.done()
        if cfg.run_id is not None and not _RUN_ID_RE.match(cfg.run_id):
            raise ConfigError(
                f"run.run_id: {cfg.run_id!r} is not a valid directory name "
                "(use letters, digits, dot, dash, underscore; must not start with '.' or '-')"
            )
        if cfg.overwrite and cfg.resume:
            raise ConfigError(
                "run: 'overwrite' and 'resume' cannot both be true — "
                "overwrite discards the run directory that resume would reuse"
            )
        return cfg

    def to_dict(self) -> dict:
        return {
            "output_root": str(self.output_root),
            "run_id": self.run_id,
            "seed": self.seed,
            "workers": self.workers,
            "overwrite": self.overwrite,
            "resume": self.resume,
        }


@dataclass
class Config:
    """A fully-parsed config file.

    `params` stays a raw mapping — each experiment parses its own params with
    its own :class:`Reader`, so adding an experiment never touches this module.
    """

    experiment: str
    run: RunConfig
    video: VideoConfig
    params: dict = field(default_factory=dict)
    source_path: Path | None = None

    KNOWN_TOP_LEVEL = ("experiment", "run", "video", "params")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Mapping, source_path: Path | None = None) -> "Config":
        r = Reader(data, path="")
        experiment = r.str("experiment")
        run = RunConfig.from_reader(r.reader("run"))
        video = VideoConfig.from_reader(r.reader("video"))
        params = r.raw("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise ConfigError(f"params: expected a mapping, got {_typename(params)}")
        r.done()
        return cls(
            experiment=experiment,
            run=run,
            video=video,
            params=dict(params),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path, overrides: Mapping | None = None) -> "Config":
        """Load and validate a YAML config, applying optional deep overrides."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        data = load_yaml(path)
        if overrides:
            data = deep_merge(data, overrides)
        return cls.from_mapping(data, source_path=path)

    # -- derived values ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "run": self.run.to_dict(),
            "video": self.video.to_dict(),
            "params": copy.deepcopy(self.params),
        }

    def config_hash(self) -> str:
        """Stable hash of everything that affects output content.

        `run_id`, `workers`, `overwrite` and `resume` are excluded: they change
        where output lands or how fast it is produced, never what it contains.
        """
        payload = self.to_dict()
        for volatile in ("run_id", "workers", "overwrite", "resume"):
            payload["run"].pop(volatile, None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def resolve_run_id(self, now: _dt.datetime | None = None) -> str:
        """Return the configured run id, or mint a timestamped one."""
        if self.run.run_id:
            return self.run.run_id
        now = now or _dt.datetime.now()
        return f"{now:%Y%m%d-%H%M%S}-{self.config_hash()}"

    def flatten(self) -> dict:
        """Flatten the whole config into scalar `cfg_*` columns for the manifest."""
        return {f"cfg_{k}": v for k, v in _flatten(self.to_dict()).items()}

    def write_snapshot(self, path: str | Path) -> None:
        """Write the resolved config beside the generated data."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_meta"] = {
            "config_hash": self.config_hash(),
            "source_path": str(self.source_path) if self.source_path else None,
            "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


# ──────────────────────────────────────────────────────────────────────────────
# YAML helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file, guaranteeing a mapping at the top level."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML — {exc}") from exc
    if data is None:
        raise ConfigError(f"{path}: file is empty")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {_typename(data)}")
    return dict(data)


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Mappings merge key-by-key; every other type (including lists) is replaced
    wholesale — a partially-overridden list of durations would be a footgun.
    """
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> dict:
    """Parse a ``dotted.key=value`` CLI override into a nested dict.

    The value is parsed as YAML, so ``run.workers=4`` yields an int and
    ``video.tune=null`` yields None.
    """
    if "=" not in text:
        raise ConfigError(f"override {text!r}: expected the form key.path=value")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r}: empty key")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {text!r}: invalid value — {exc}") from exc

    out: dict = {}
    cursor = out
    parts = key.split(".")
    for part in parts[:-1]:
        if not part:
            raise ConfigError(f"override {text!r}: empty path segment")
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return out


def _flatten(data: Any, prefix: str = "") -> dict:
    """Flatten nested mappings to dotted keys; lists become JSON strings."""
    out: dict = {}
    if isinstance(data, Mapping):
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, child))
    elif isinstance(data, (list, tuple)):
        out[prefix] = json.dumps(list(data), default=str)
    else:
        out[prefix] = data
    return out
