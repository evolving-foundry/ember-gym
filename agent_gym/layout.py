"""Single source of truth for AgentGym overlay, repository, workspace and data roots."""

from __future__ import annotations

import importlib.util
import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ENV_ROOT = "AGENT_GYM_ROOT"
ENV_REPO = "AGENT_GYM_REPO"
ENV_OVERLAY = "AGENT_GYM_OVERLAY"
ENV_WORKSPACE = "AGENT_GYM_WORKSPACE"
ENV_DATA_ROOT = "AGENT_GYM_DATA_ROOT"
ENV_EXTRA_ROOTS = "AGENT_GYM_EXTRA_ROOTS"
ENV_NEMO_EXTRA_ROOTS = "NEMO_GYM_EXTRA_ROOTS"
OVERLAY_MANIFEST = "overlay-manifest.yaml"


class LayoutError(RuntimeError):
    """Raised when a layout root or relative path cannot be resolved safely."""


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and base is not None:
        value = base / value
    return value.resolve(strict=False)


def _split_roots(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item) for item in value.split(os.pathsep) if item]


def _dedupe(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return tuple(result)


def _framework_root() -> Path:
    spec = importlib.util.find_spec("nemo_gym")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve(strict=False).parent
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve(strict=False).parent.parent
    for key in ("purelib", "platlib"):
        candidate = Path(sysconfig.get_paths()[key])
        if candidate.is_dir():
            return candidate.resolve(strict=False)
    return Path(sysconfig.get_paths()["purelib"]).resolve(strict=False)


def _find_layout_root(start: Path) -> Path | None:
    """Find the nearest source checkout or deployment root from a nested cwd.

    A release prefix may be nested under the source checkout (for example under ``.tmp``).
    Searching for source markers first would then resolve the ancestor checkout and silently
    ignore the deployment's own overlay. The overlay manifest is the unambiguous nearest-root
    marker shared by both source and release layouts.
    """
    for candidate in (start, *start.parents):
        if (candidate / "overlay" / OVERLAY_MANIFEST).is_file():
            return candidate.resolve(strict=False)
    return None


@dataclass(frozen=True)
class GymLayout:
    """Resolved roots shared by CLI, eval_service, manifest and dataset loading.

    Search order for code/config is overlay -> explicit extra roots -> repo -> workspace -> wheel.
    Dataset reads use ``resolve_data`` and consult ``data_root`` first. Roots are never inferred from
    whether a local copy happens to exist; the order is stable for every consumer.
    """

    repo_root: Path
    overlay_root: Path
    workspace_root: Path
    data_root: Path | None
    extra_roots: tuple[Path, ...]
    framework_root: Path

    @classmethod
    def discover(
        cls,
        *,
        repo_root: str | Path | None = None,
        overlay_root: str | Path | None = None,
        workspace_root: str | Path | None = None,
        data_root: str | Path | None = None,
        extra_roots: Sequence[str | Path] | None = None,
        framework_root: str | Path | None = None,
    ) -> "GymLayout":
        cwd = Path.cwd().resolve(strict=False)
        explicit_repo = repo_root or os.environ.get(ENV_REPO) or os.environ.get(ENV_ROOT)
        overlay_value = overlay_root or os.environ.get(ENV_OVERLAY)
        discovered_root = _find_layout_root(cwd) if explicit_repo is None else None

        if explicit_repo is not None:
            repo = _resolve_path(explicit_repo, base=cwd)
        elif overlay_value is not None and Path(overlay_value).expanduser().is_absolute():
            # C-12: an installed deployment may expose only AGENT_GYM_OVERLAY. Its parent is
            # the deployment/repository root used for identity, data and search roots.
            repo = _resolve_path(Path(overlay_value).expanduser()).parent
        elif discovered_root is not None:
            repo = discovered_root
        else:
            repo = cwd

        if overlay_value is not None:
            overlay_base = repo if explicit_repo is not None or discovered_root is not None else cwd
            overlay = _resolve_path(overlay_value, base=overlay_base)
            if explicit_repo is None:
                # Keep the only-overlay contract even when a relative overlay path was given.
                repo = overlay.parent
        else:
            overlay = repo / "overlay"
        workspace_value = workspace_root or os.environ.get(ENV_WORKSPACE)
        workspace = _resolve_path(workspace_value, base=repo) if workspace_value else repo
        data_value = data_root if data_root is not None else os.environ.get(ENV_DATA_ROOT)
        data = _resolve_path(data_value, base=workspace) if data_value else workspace / "data"
        configured_extras: list[Path] = []
        if extra_roots:
            configured_extras.extend(Path(item) for item in extra_roots)
        configured_extras.extend(_split_roots(os.environ.get(ENV_EXTRA_ROOTS)))
        configured_extras.extend(_split_roots(os.environ.get(ENV_NEMO_EXTRA_ROOTS)))
        extras = tuple(_resolve_path(item, base=repo) for item in configured_extras)
        framework = _resolve_path(framework_root) if framework_root else _framework_root()
        return cls(
            repo_root=repo,
            overlay_root=overlay,
            workspace_root=workspace,
            data_root=data,
            extra_roots=_dedupe(extras),
            framework_root=framework,
        )

    @property
    def search_roots(self) -> tuple[Path, ...]:
        return _dedupe(
            [
                self.overlay_root,
                *self.extra_roots,
                self.repo_root,
                self.workspace_root,
                self.framework_root,
            ]
        )

    @property
    def data_roots(self) -> tuple[Path, ...]:
        roots = list(self.search_roots)
        if self.data_root is not None:
            roots.insert(0, self.data_root)
        return _dedupe(roots)

    @property
    def manifest_path(self) -> Path:
        return self.overlay_root / OVERLAY_MANIFEST

    def require_overlay(self) -> None:
        if not self.overlay_root.is_dir():
            raise LayoutError(f"overlay root does not exist: {self.overlay_root}; set {ENV_OVERLAY}")
        if not self.manifest_path.is_file():
            raise LayoutError(f"overlay manifest does not exist: {self.manifest_path}")

    def _resolve_in_roots(
        self,
        fpath: str | Path,
        roots: Sequence[Path],
        *,
        must_exist: bool,
        validator: Callable[[Path], bool] | None,
    ) -> Path:
        candidate_path = Path(fpath).expanduser()
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve(strict=False)
            if must_exist and not resolved.exists():
                raise LayoutError(f"path does not exist: {resolved}")
            return resolved
        check = validator or Path.exists
        candidates: list[Path] = []
        for root in roots:
            candidate = (root / candidate_path).resolve(strict=False)
            candidates.append(candidate)
            if check(candidate):
                return candidate
        if not must_exist:
            return candidates[0]
        searched = "\n".join(f"  - {path}" for path in candidates)
        raise LayoutError(f"relative path {fpath!r} not found under any root:\n{searched}")

    def resolve(
        self,
        fpath: str | Path,
        *,
        must_exist: bool = True,
        validator: Callable[[Path], bool] | None = None,
    ) -> Path:
        """Resolve a component/config/read path using the stable code search order."""
        return self._resolve_in_roots(fpath, self.search_roots, must_exist=must_exist, validator=validator)

    def resolve_data(
        self,
        fpath: str | Path,
        *,
        must_exist: bool = True,
        validator: Callable[[Path], bool] | None = None,
    ) -> Path:
        """Resolve dataset/holdout/jsonl paths; data_root wins over code roots."""
        return self._resolve_in_roots(fpath, self.data_roots, must_exist=must_exist, validator=validator)

    def resolve_write(self, fpath: str | Path, *, base: Path | None = None) -> Path:
        """Resolve a relative write target once; absolute caller paths stay absolute."""
        candidate = Path(fpath).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        return ((base or self.workspace_root) / candidate).resolve(strict=False)

    def component_roots(self) -> tuple[Path, ...]:
        """Roots searched by Gym discovery, excluding the wheel fallback."""
        return _dedupe([self.overlay_root, *self.extra_roots, self.repo_root, self.workspace_root])

    def shadowing_errors(self) -> list[str]:
        errors: list[str] = []
        for root in self.component_roots():
            candidate = root / "nemo_gym"
            has_files = candidate.is_dir() and any(path.is_file() for path in candidate.rglob("*"))
            if candidate.is_file() or has_files:
                errors.append(f"framework package shadowed under component root: {candidate}")
        return errors

    def assert_framework_not_shadowed(self) -> None:
        errors = self.shadowing_errors()
        if errors:
            raise LayoutError("\n".join(errors))

    def gym_env(self, *, environ: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if environ is None else environ)
        extra = _dedupe([self.overlay_root, *self.extra_roots, self.repo_root, self.workspace_root])
        env[ENV_ROOT] = str(self.repo_root)
        env[ENV_REPO] = str(self.repo_root)
        env[ENV_OVERLAY] = str(self.overlay_root)
        env[ENV_WORKSPACE] = str(self.workspace_root)
        env[ENV_NEMO_EXTRA_ROOTS] = os.pathsep.join(str(path) for path in extra)
        if self.data_root is not None:
            env[ENV_DATA_ROOT] = str(self.data_root)
        return env
