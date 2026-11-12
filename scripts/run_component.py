#!/usr/bin/env python3
"""Run a component script from the writable prepare staging tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_gym.bootstrap import prepare_runtime
from agent_gym.staging import cleanup_staged_components, stage_prepare_components


COMPONENT_ROOTS = frozenset(
    {
        "benchmarks",
        "resources_servers",
        "responses_api_agents",
        "responses_api_models",
        "environment_servers",
    }
)


def _component_script(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0] not in COMPONENT_ROOTS
    ):
        raise ValueError("script must be a repository-relative component path such as benchmarks/osworld/prepare.py")
    return relative


def _command(script: Path, arguments: list[str]) -> list[str]:
    suffix = script.suffix.lower()
    if suffix == ".py":
        return [sys.executable, str(script), *arguments]
    if suffix == ".sh":
        return ["bash", str(script), *arguments]
    return [str(script), *arguments]


def usage() -> str:
    return (
        "usage: python scripts/run_component.py COMPONENT_SCRIPT [ARG ...]\n\n"
        "Run a Python or shell script from the merged component tree in the writable\n"
        "workspace. Use the logical path (benchmarks/<name>/...), not overlay/... .\n"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        stream = sys.stdout if arguments else sys.stderr
        print(usage(), file=stream, end="")
        return 0 if arguments else 2

    try:
        relative = _component_script(arguments[0])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    layout = prepare_runtime(
        require_overlay=True,
        verify_trees=True,
        chdir=False,
        import_framework=False,
    )
    workspace = stage_prepare_components(layout)
    script = workspace / relative
    try:
        if not script.is_file():
            print(f"error: staged component script not found: {relative}", file=sys.stderr)
            return 2
        environment = os.environ.copy()
        pythonpath = [str(workspace), environment.get("PYTHONPATH", "")]
        environment["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
        return subprocess.run(
            _command(script, arguments[1:]),
            cwd=workspace,
            env=environment,
            check=False,
        ).returncode
    finally:
        cleanup_staged_components(layout)


if __name__ == "__main__":
    raise SystemExit(main())
