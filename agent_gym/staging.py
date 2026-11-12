"""Materialize a writable component view for legacy data-preparation commands."""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Iterator

from agent_gym.layout import GymLayout


COMPONENT_DIRECTORIES = (
    "benchmarks",
    "resources_servers",
    "responses_api_agents",
    "responses_api_models",
    "environment_servers",
)
RUNTIME_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "cache",
        "data",
        "node_modules",
        "outputs",
        "results",
        "tau2_data",
        "venv",
    }
)
STAGE_MANIFEST = Path(".agent-gym") / "prepare-stage.json"


class ComponentStageError(RuntimeError):
    """Raised when the prepare-time component view cannot be materialized safely."""


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield code files while excluding data, virtualenvs and runtime caches."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise ComponentStageError(f"cannot list component directory {directory}: {exc}") from exc
        for entry in entries:
            if entry.name in RUNTIME_DIRECTORIES:
                continue
            if entry.is_symlink():
                if entry.is_file():
                    yield entry
                continue
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file():
                yield entry


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _copy_file(source: Path, destination: Path, *, overwrite: bool) -> bool:
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        return False
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.agent-gym-stage")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        raise ComponentStageError(f"cannot stage {source} -> {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _load_managed(workspace_root: Path) -> set[str]:
    path = workspace_root / STAGE_MANIFEST
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    files = payload.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        return set()
    return {item for item in files if _is_safe_relative_path(item)}


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _write_managed(workspace_root: Path, files: set[str]) -> None:
    path = workspace_root / STAGE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "files": sorted(files)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stage_prepare_components(layout: GymLayout) -> Path:
    """Copy the merged component code view into the writable workspace.

    Upstream's ``eval prepare`` checks and imports ``benchmarks/<name>/prepare.py`` as a
    cwd-relative path. In a wheel deployment that path lives under the framework wheel or the
    overlay, so the launcher's workspace ``chdir`` makes it appear missing. The staged copy keeps
    the upstream relative-path contract and lets prepare scripts write to ``workspace/benchmarks``
    without mutating the integrity-checked overlay.
    """
    workspace = layout.workspace_root.resolve(strict=False)
    workspace.mkdir(parents=True, exist_ok=True)
    managed = _load_managed(workspace)
    current_sources: set[str] = set()
    written: set[str] = set()

    # Layout precedence is overlay > extras > repo > workspace > wheel. Apply the inverse so the
    # staged tree has the same effective contents while the workspace remains the write target.
    for root in reversed(layout.search_roots):
        if _same_path(root, workspace):
            continue
        is_wheel = _same_path(root, layout.framework_root)
        for directory_name in COMPONENT_DIRECTORIES:
            source_root = root / directory_name
            if not source_root.is_dir():
                continue
            for source in _iter_files(source_root):
                relative = source.relative_to(root).as_posix()
                destination = workspace / relative
                current_sources.add(relative)
                overwrite = not is_wheel or relative in managed
                if _copy_file(source, destination, overwrite=overwrite):
                    written.add(relative)

    retained = managed & current_sources
    for relative in sorted(managed - current_sources, reverse=True):
        stale = workspace / relative
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            raise ComponentStageError(f"cannot remove stale staged component {stale}: {exc}") from exc
    _write_managed(workspace, written | retained)
    return workspace


def cleanup_staged_components(layout: GymLayout) -> int:
    """Remove prepare-time code copies while preserving data and generated artifacts.

    The staged tree exists only because upstream resolves ``prepare_script`` relative to
    the process cwd. Keeping it after the command would make the workspace shadow the
    wheel during later ``eval run`` calls, so a wheel upgrade could silently continue
    executing stale code. Generated files not recorded in the manifest (most notably
    ``data/**``) are deliberately retained.
    """
    workspace = layout.workspace_root.resolve(strict=False)
    managed = _load_managed(workspace)
    removed = 0
    staged_directories: set[Path] = set()
    for relative in sorted(managed, reverse=True):
        staged = workspace / relative
        directory = staged.parent
        while directory != workspace and workspace in directory.parents:
            staged_directories.add(directory)
            directory = directory.parent
        try:
            if staged.is_file() or staged.is_symlink():
                staged.unlink()
                removed += 1
        except OSError as exc:
            raise ComponentStageError(f"cannot remove staged component {staged}: {exc}") from exc

    manifest = workspace / STAGE_MANIFEST
    try:
        manifest.unlink(missing_ok=True)
    except OSError as exc:
        raise ComponentStageError(f"cannot remove staging manifest {manifest}: {exc}") from exc
    _prune_staged_directories(staged_directories)
    return removed


def _prune_staged_directories(directories: set[Path]) -> None:
    """Remove import caches and now-empty directories touched by staging."""
    ordered = sorted(directories, key=lambda path: len(path.parts), reverse=True)
    for directory in ordered:
        if directory.is_symlink():
            continue
        cache = directory / "__pycache__"
        if cache.is_dir() and not cache.is_symlink():
            try:
                shutil.rmtree(cache)
            except OSError as exc:
                raise ComponentStageError(f"cannot remove staged cache {cache}: {exc}") from exc
    for directory in ordered:
        if directory.is_symlink():
            continue
        try:
            directory.rmdir()
        except OSError:
            # Preserve directories that still contain generated or user-owned files.
            pass


def activate_staged_components(layout: GymLayout) -> None:
    """Put the staged workspace first on ``sys.path`` without changing config precedence.

    Upstream uses component search roots for both import and config discovery. Reordering the
    environment would make every staged config look like a duplicate overlay config. Moving only
    the ``sys.path`` entry keeps discovery overlay-first while forcing imports of relative
    ``prepare_script`` modules to the writable staged copy.
    """
    workspace = layout.workspace_root.resolve(strict=False)
    retained: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve(strict=False)
        except OSError:
            retained.append(entry)
            continue
        if resolved != workspace:
            retained.append(entry)
    sys.path[:] = [str(workspace), *retained]
