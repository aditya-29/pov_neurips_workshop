"""The repo must be self-contained: a fresh clone is the whole thing.

This file exists because it was not. A bare ``data/`` line in .gitignore matches
a directory of that name at *any* depth, so ``pov/prompts/data/`` — every task
prompt in the benchmark — was silently never committed. The suite passed anyway,
because on the machine where it was written the files were sitting there
untracked. Anyone cloning the repo got a package that raised on import of its
own prompts.

So these tests do not ask "does it work here". They ask "does it work for
someone who has nothing but this clone":

  * every runtime file the package needs is tracked by git,
  * nothing under pov/ is swallowed by an ignore rule,
  * no module reaches outside the repo by absolute path,
  * pov imports using only its declared dependencies.

They are cheap and they run everywhere. Keep them passing.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "pov"


def git(*args: str) -> str:
    """Run git in the repo and return stdout."""
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture(scope="module")
def in_git_repo() -> bool:
    try:
        git("rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pytest.skip("not a git checkout")
    return True


@pytest.fixture(scope="module")
def tracked(in_git_repo) -> set[str]:
    """Every path git knows about, repo-relative and POSIX-style."""
    return {line for line in git("ls-files", "-z").split("\0") if line}


# ── Nothing the package needs may be missing from a clone ─────────────────────


class TestEverythingRuntimeIsTracked:
    def test_every_file_under_pov_is_tracked(self, tracked):
        """No untracked file may live inside the importable package.

        An untracked file under pov/ is either dead weight or — as with the
        prompts — a runtime dependency that a clone will not have.
        """
        on_disk = {
            path.relative_to(REPO).as_posix()
            for path in PACKAGE.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not path.name.endswith(".pyc")
        }
        missing = sorted(on_disk - tracked)
        assert not missing, (
            "these files are inside the package but absent from a fresh clone:\n  "
            + "\n  ".join(missing)
        )

    def test_no_ignore_rule_matches_anything_under_pov(self, in_git_repo):
        """The direct check for the bug that motivated this file."""
        files = [
            str(path.relative_to(REPO))
            for path in PACKAGE.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        # check-ignore exits 1 when nothing matches, which is what we want.
        result = subprocess.run(
            ["git", "check-ignore", "--stdin", "-v"],
            cwd=REPO, input="\n".join(files), capture_output=True, text=True,
        )
        assert not result.stdout.strip(), (
            "ignore rules are hiding package files (rule -> file):\n"
            + result.stdout
        )

    def test_every_registered_prompt_is_tracked(self, tracked):
        from pov import prompts

        missing = [
            filename
            for filename in sorted(set(prompts._REGISTRY.values()))
            if f"pov/prompts/data/{filename}" not in tracked
        ]
        assert not missing, f"prompts missing from the repo: {missing}"

    def test_package_data_globs_resolve(self, tracked):
        """Every pyproject package-data pattern must match tracked files.

        A pattern that matches nothing means the wheel ships without them.
        """
        config = tomllib.loads((REPO / "pyproject.toml").read_text())
        package_data = config["tool"]["setuptools"]["package-data"]
        for package, patterns in package_data.items():
            directory = REPO / package.replace(".", "/")
            for pattern in patterns:
                matches = [
                    p.relative_to(REPO).as_posix() for p in directory.glob(pattern)
                ]
                assert matches, f"package-data {package}:{pattern} matches nothing"
                untracked = [m for m in matches if m not in tracked]
                assert not untracked, (
                    f"package-data {package}:{pattern} matches untracked files: "
                    f"{untracked}"
                )

    @pytest.mark.parametrize(
        "path",
        [
            "configs/chess.yaml",
            "configs/asl.yaml",
            "configs/wbw_mcq.yaml",
            "examples/questions.jsonl",
            "scripts/fetch_mmlu.py",
            "scripts/download_how2sign.sh",
            "README.md",
            "pyproject.toml",
        ],
    )
    def test_documented_entry_points_are_tracked(self, tracked, path):
        """Everything the README tells a newcomer to run must be in the clone."""
        assert path in tracked, f"{path} is referenced but not committed"


# ── No path may reach outside the repo ────────────────────────────────────────


class TestNoExternalPaths:
    #: Directories whose sources must not hardcode machine-specific paths.
    SOURCE_DIRS = ("pov", "scripts", "configs", "tests", "docs")

    #: This file necessarily contains the patterns it searches for.
    SELF = Path(__file__).resolve()

    def _source_files(self):
        for directory in self.SOURCE_DIRS:
            root = REPO / directory
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if path.resolve() == self.SELF:
                    continue
                if path.suffix in {".py", ".yaml", ".yml", ".sh", ".toml", ".md",
                                   ".jsonl", ".txt", ".cfg"}:
                    yield path

    def test_no_source_file_hardcodes_a_home_directory(self):
        """`/Users/...`, `/home/...`, `C:\\Users\\...` are machine-specific.

        The one legitimate use — checking the original study for prompt
        provenance — is opt-in through the POV_REFERENCE_REPO environment
        variable, so it does not appear as a literal.
        """
        offenders = []
        for path in self._source_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                home_like = "/Users/" in line or "/home/" in line
                if home_like and "/home/runner" not in line:
                    offenders.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
        assert not offenders, (
            "machine-specific paths in tracked sources:\n  " + "\n  ".join(offenders)
        )

    def test_no_source_file_references_the_original_study_by_path(self):
        """The prompts are vendored; the icml_workshop checkout is not needed."""
        offenders = []
        for path in self._source_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if "icml_workshop" not in line:
                    continue
                # Prose and the opt-in env-var docs may name it; a path may not.
                if "/" in line.split("icml_workshop")[0].strip().rstrip("/"):
                    stripped = line.strip()
                    if not stripped.startswith("#"):
                        offenders.append(
                            f"{path.relative_to(REPO)}:{number}: {stripped}"
                        )
        assert not offenders, (
            "sources still point at the original repo:\n  " + "\n  ".join(offenders)
        )

    def test_config_paths_stay_inside_the_repo(self):
        """Shipped configs must use relative paths under the repo."""
        import yaml

        for path in (REPO / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            for key, value in (config.get("params") or {}).items():
                if isinstance(value, str) and ("/" in value or "\\" in value):
                    if value.startswith(("http://", "https://")):
                        continue
                    assert not Path(value).is_absolute(), (
                        f"{path.name}: params.{key} is an absolute path: {value}"
                    )
            output_root = (config.get("run") or {}).get("output_root")
            if output_root:
                assert not Path(output_root).is_absolute(), (
                    f"{path.name}: run.output_root is absolute: {output_root}"
                )


# ── The package must import on its declared dependencies alone ────────────────


class TestDeclaredDependencies:
    #: Distribution name -> the name you actually import.
    IMPORT_NAMES = {
        "pyyaml": "yaml",
        "pillow": "PIL",
        "python-chess": "chess",
    }

    @pytest.fixture(scope="class")
    @classmethod
    def declared(cls) -> dict[str, set[str]]:
        config = tomllib.loads((REPO / "pyproject.toml").read_text())
        project = config["project"]

        def names(requirements):
            out = set()
            for requirement in requirements:
                dist = (
                    requirement.split(";")[0]
                    .split("[")[0]
                    .split("=")[0]
                    .split(">")[0]
                    .split("<")[0]
                    .split("!")[0]
                    .strip()
                    .lower()
                )
                out.add(cls.IMPORT_NAMES.get(dist, dist.replace("-", "_")))
            return out

        optional = project.get("optional-dependencies", {})
        return {
            "required": names(project["dependencies"]),
            "optional": set().union(*(names(v) for v in optional.values())),
        }

    def _top_level_imports(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
        return found

    def test_pov_imports_nothing_undeclared(self, declared):
        """Every third-party import in pov/ is a declared dependency."""
        stdlib = sys.stdlib_module_names
        allowed = stdlib | declared["required"] | declared["optional"] | {"pov"}

        offenders: dict[str, set[str]] = {}
        for path in PACKAGE.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            extra = self._top_level_imports(path) - allowed
            if extra:
                offenders[path.relative_to(REPO).as_posix()] = extra
        assert not offenders, f"undeclared imports: {offenders}"

    def test_core_generation_needs_no_optional_dependency(self, declared):
        """Import pov and generate without chess-svg / datasets / gdown.

        Those are conveniences. If a core module imports one unguarded, a
        minimal install breaks.
        """
        stdlib = sys.stdlib_module_names
        allowed = stdlib | declared["required"] | {"pov"}

        core = [
            PACKAGE / "cli.py",
            PACKAGE / "config.py",
            PACKAGE / "manifest.py",
            PACKAGE / "video.py",
            PACKAGE / "layout.py",
            PACKAGE / "registry.py",
        ]
        for path in core:
            extra = self._top_level_imports(path) - allowed
            assert not extra, (
                f"{path.name} imports optional dependencies at module level: {extra}"
            )

    def test_every_optional_import_is_guarded(self):
        """Optional dependencies must be imported inside try/except or a function.

        A bare module-level `import cairosvg` would make the whole package
        unimportable on a minimal install.
        """
        optional_modules = {"chess", "cairosvg", "datasets", "gdown"}
        offenders = []
        for path in PACKAGE.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = {node.module.split(".")[0]}
                else:
                    continue
                bad = names & optional_modules
                if bad:
                    offenders.append(f"{path.relative_to(REPO)}: {sorted(bad)}")
        assert not offenders, (
            "unguarded module-level imports of optional dependencies:\n  "
            + "\n  ".join(offenders)
        )


# ── A clone must actually work ────────────────────────────────────────────────


class TestCloneWorks:
    def test_prompts_load_from_the_installed_package(self):
        """Read prompts the way an installed wheel does, not off the source tree."""
        from pov import prompts

        for experiment, kind in prompts.available():
            assert prompts.get(experiment, kind).strip()

    def test_the_package_imports_with_the_repo_off_sys_path(self, tmp_path):
        """Import pov from another directory — catches accidental relative paths.

        Run in a subprocess from tmp_path so a stray `Path("configs/...")` or a
        cwd-relative default fails here rather than in someone's clone.
        """
        code = (
            "import pov, pov.cli, pov.manifest, pov.video, pov.eval, pov.prompts;"
            "from pov.registry import EXPERIMENTS;"
            "assert set(EXPERIMENTS) == {'chess', 'asl', 'wbw_mcq'}, EXPERIMENTS;"
            "assert pov.prompts.get('chess').strip();"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_cli_help_works_from_anywhere(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "pov.cli", "--help"],
            cwd=tmp_path, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        assert result.returncode == 0, result.stderr
