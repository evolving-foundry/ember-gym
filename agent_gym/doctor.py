"""Installation, entry-point and overlay diagnostics for AgentGym Gym."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from agent_gym.identity import configuration_identity
from agent_gym.layout import GymLayout, LayoutError
from agent_gym.overlay import OverlayError, verify_manifest


PROBE_ENV = "AGENT_GYM_ENTRYPOINT_PROBE"
PROBE_PREFIX = "agent-gym:"


def _expected_entrypoints() -> list[str]:
    try:
        dist = importlib.metadata.distribution("nemo-gym")
    except importlib.metadata.PackageNotFoundError as exc:
        raise LayoutError("nemo-gym distribution is not installed") from exc
    names = sorted(entry.name for entry in dist.entry_points if entry.group == "console_scripts")
    if not names:
        raise LayoutError("nemo-gym distribution declares no console scripts")
    return sorted(set(names) | {"agent-gym"})


def _entrypoint_script(name: str) -> str | None:
    """Prefer console scripts installed beside the active interpreter."""

    interpreter_dir = Path(sys.executable).parent
    return shutil.which(name, path=str(interpreter_dir)) or shutil.which(name)


def _probe_entrypoint(name: str, *, timeout: float = 10.0) -> str | None:
    script = _entrypoint_script(name)
    if script is None:
        return f"{name}: script not found on PATH"
    env = dict(os.environ)
    env[PROBE_ENV] = "1"
    try:
        proc = subprocess.run(
            [script, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{name}: probe failed: {exc}"
    expected = f"{PROBE_PREFIX}{name}"
    actual = proc.stdout.strip()
    if proc.returncode != 0:
        return f"{name}: probe exited {proc.returncode}: {proc.stderr.strip()}"
    if actual != expected:
        return f"{name}: expected {expected!r}, got {actual!r}"
    return None


def check_entrypoints() -> list[str]:
    errors: list[str] = []
    for name in _expected_entrypoints():
        error = _probe_entrypoint(name)
        if error is not None:
            errors.append(error)
    return errors


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None, *, layout: GymLayout | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-gym doctor")
    parser.add_argument("--entrypoints", action="store_true", help="probe every installed CLI alias")
    parser.add_argument("--overlay", action="store_true", help="verify manifest and overlay tree hashes")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(list(argv) if argv is not None else None)
    active_layout = layout or GymLayout.discover()
    errors: list[str] = []
    payload: dict[str, Any] = {
        "layout": {
            "repo_root": str(active_layout.repo_root),
            "overlay_root": str(active_layout.overlay_root),
            "workspace_root": str(active_layout.workspace_root),
            "data_root": str(active_layout.data_root) if active_layout.data_root else None,
            "framework_root": str(active_layout.framework_root),
        }
    }
    try:
        payload["identity"] = configuration_identity(active_layout)
    except (LayoutError, OverlayError, OSError) as exc:
        errors.append(str(exc))
    if args.entrypoints:
        entrypoint_errors = check_entrypoints()
        payload["entrypoints"] = {"ok": not entrypoint_errors, "errors": entrypoint_errors}
        errors.extend(entrypoint_errors)
    if args.overlay:
        try:
            manifest = verify_manifest(active_layout)
            payload["overlay"] = {
                "ok": True,
                "overlay_version": manifest.get("overlay_version"),
                "components": sorted((manifest.get("components") or {}).keys()),
                "assets": sorted((manifest.get("assets") or {}).keys()),
                "tools": sorted((manifest.get("tools") or {}).keys()),
            }
        except (LayoutError, OverlayError, OSError, ValueError) as exc:
            message = f"overlay: {exc}"
            errors.append(message)
            payload["overlay"] = {"ok": False, "errors": [message]}
    if errors:
        payload["ok"] = False
        payload["errors"] = errors
        if args.json:
            _print_json(payload)
        else:
            for error in errors:
                print(f"ERROR: {error}")
        return 1
    payload["ok"] = True
    if args.json:
        _print_json(payload)
    else:
        print("agent-gym doctor: ok")
        if args.entrypoints:
            print(f"entrypoints: {len(_expected_entrypoints())} aliases verified")
        if args.overlay:
            print("overlay: manifest and tree hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
