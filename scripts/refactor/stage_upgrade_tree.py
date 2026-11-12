"""Stage the effective tree for a nemo-gym wheel upgrade.

The result is the old wheel tree with the current overlay applied. Pass it to
audit_tree.py together with the old and new wheel roots and
``--baseline-mode wheel``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".baseline",
    "cache",
    "results",
    "outputs",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED_DIRS or name.endswith(".dist-info")}


def stage(old_wheel: Path, overlay: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_wheel, destination, dirs_exist_ok=True, ignore=_ignore)
    shutil.copytree(overlay, destination, dirs_exist_ok=True, ignore=_ignore)
    manifest = destination / "overlay-manifest.yaml"
    if manifest.exists():
        manifest.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-wheel", type=Path, required=True, help="extracted previously pinned wheel")
    parser.add_argument("--overlay", type=Path, required=True, help="current AgentGym overlay root")
    parser.add_argument("--output", type=Path, required=True, help="empty destination directory")
    args = parser.parse_args()
    stage(args.old_wheel.resolve(), args.overlay.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
