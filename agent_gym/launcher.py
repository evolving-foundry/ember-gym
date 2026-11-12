"""Compatibility launcher for agent-gym, gym, ng and all legacy console scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agent_gym.bootstrap import prepare_runtime, shadowing_component_root
from agent_gym.layout import GymLayout, LayoutError
from agent_gym.staging import (
    activate_staged_components,
    cleanup_staged_components,
    stage_prepare_components,
)


PROBE_ENV = "AGENT_GYM_ENTRYPOINT_PROBE"
_CANONICAL = {"agent-gym", "gym", "ng"}
_LEGACY_PREFIXES = ("nemo_gym_", "ng_")
_LEGACY_PREPARE = {"nemo_gym_prepare_benchmark", "ng_prepare_benchmark"}
_DIAGNOSTIC_FLAGS = {"-h", "--help", "--version"}
_LAYOUT_READ_FLAGS = {"--benchmark-config", "--config"}
_PATH_FLAGS = {
    "-c",
    "-i",
    "-o",
    "--atif-manifest",
    "--benchmark-config",
    "--config",
    "--data-root",
    "--dataset",
    "--dir",
    "--input",
    "--input-glob",
    "--inputs",
    "--journal-path",
    "--ledger-path",
    "--manifest",
    "--output",
    "--output-dir",
    "--overlay-root",
    "--path",
    "--profile-path",
    "--reports-root",
    "--repo-root",
    "--results-root",
    "--rollouts",
    "--venv",
    "--workdir",
    "--workspace-root",
}


def _entrypoint_name() -> str:
    return Path(sys.argv[0]).stem


_shadowing_component_root = shadowing_component_root


def _is_diagnostic_invocation(argv: list[str]) -> bool:
    return any(token in _DIAGNOSTIC_FLAGS for token in argv[1:])


def _needs_prepare_staging(name: str, argv: list[str]) -> bool:
    if _is_diagnostic_invocation(argv):
        return False
    return (name in _CANONICAL and argv[1:3] == ["eval", "prepare"]) or name in _LEGACY_PREPARE


def _absolute_path_argument(value: str, caller_cwd: Path) -> str:
    if value == "-" or not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return value
    return str((caller_cwd / path).resolve(strict=False))


def _resolve_layout_read_path(value: str, caller_cwd: Path, layout: GymLayout | None) -> str:
    """Keep caller-relative config paths, but fall back to overlay/repo/workspace/wheel."""
    candidate = Path(_absolute_path_argument(value, caller_cwd))
    if layout is None or candidate.exists():
        return str(candidate)
    try:
        return str(layout.resolve(value))
    except LayoutError:
        # Preserve the upstream error path and message when no root contains the config.
        return str(candidate)


def _absolutize_path_arguments(argv: list[str], caller_cwd: Path, *, layout: GymLayout | None = None) -> list[str]:
    """Preserve legacy relative path semantics before the workspace chdir.

    The upstream CLI parses paths after import. Resolving documented path options against the
    caller's original cwd keeps commands invoked from a subdirectory equivalent to running them
    from that directory before the refactor.
    """
    result = list(argv)
    index = 1
    while index < len(result):
        token = result[index]
        if token == "--":
            break
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in _PATH_FLAGS:
                if flag in _LAYOUT_READ_FLAGS:
                    value = _resolve_layout_read_path(value, caller_cwd, layout)
                else:
                    value = _absolute_path_argument(value, caller_cwd)
                result[index] = f"{flag}={value}"
        elif token in _PATH_FLAGS and index + 1 < len(result):
            if token in _LAYOUT_READ_FLAGS:
                result[index + 1] = _resolve_layout_read_path(result[index + 1], caller_cwd, layout)
            else:
                result[index + 1] = _absolute_path_argument(result[index + 1], caller_cwd)
            index += 1
        index += 1
    return result


def _prepare_runtime(*, require_overlay: bool = True) -> GymLayout:
    return prepare_runtime(require_overlay=require_overlay, verify_trees=True)


def main() -> None:
    name = _entrypoint_name()
    if os.environ.get(PROBE_ENV) == "1":
        print(f"agent-gym:{name}")
        return

    if name in _CANONICAL and len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from agent_gym.doctor import main as doctor_main

        raise SystemExit(doctor_main(sys.argv[2:]))

    caller_cwd = Path.cwd().resolve(strict=False)
    layout = GymLayout.discover()
    sys.argv[:] = _absolutize_path_arguments(sys.argv, caller_cwd, layout=layout)
    diagnostic = _is_diagnostic_invocation(sys.argv)
    _prepare_runtime(require_overlay=not diagnostic)
    prepare_staging = _needs_prepare_staging(name, sys.argv)
    if not diagnostic:
        # A previous prepare may have left a stale wheel/overlay copy after a hard crash.
        cleanup_staged_components(layout)
    if prepare_staging:
        stage_prepare_components(layout)
        activate_staged_components(layout)
    try:
        if name in _CANONICAL:
            from nemo_gym.cli.main import main as upstream_main
        elif name.startswith(_LEGACY_PREFIXES):
            from nemo_gym.cli.legacy import main as upstream_main

            sys.argv[0] = name
        else:
            raise SystemExit(f"unknown AgentGym Gym entry point: {name}")

        upstream_main()
    finally:
        if prepare_staging:
            cleanup_staged_components(layout)


if __name__ == "__main__":
    main()
