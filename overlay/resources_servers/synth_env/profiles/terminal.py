# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal profile (M1, design §3.3): bash actions + text observations + a virtual FS.

The profile owns three things: the virtual filesystem (content-addressed
in-memory dict; sessions get deep copies of the package snapshot), action
parsing/validation against the package's env_spec (single ``bash`` tool call,
length cap, dangerous-command blacklist), and observation assembly (stdout/
stderr text plus a change summary against the seed baseline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from resources_servers.synth_env.dynamics.replay import ReplayOutcome
from resources_servers.synth_env.package_loader import EnvSpec


class SynthEnvActionError(ValueError):
    """An action dict that violates the profile's action schema (enum/schema/constraints)."""


class VirtualFS:
    """In-memory virtual filesystem: absolute path -> text content; directories implicit."""

    def __init__(self, files: Optional[dict[str, str]] = None):
        self.files: dict[str, str] = dict(files or {})

    def copy(self) -> "VirtualFS":
        return VirtualFS(self.files)

    def apply(self, writes: dict[str, str], deletes: list[str]) -> None:
        self.files.update(writes)
        for path in deletes:
            self.files.pop(path, None)

    def tree(self, path: str = "/", depth: int = 2) -> str:
        prefix = "" if path in ("", "/") else path.rstrip("/") + "/"
        entries = sorted(
            name for name in self.files if name.startswith(prefix)
        )
        lines = []
        for full_path in entries[: 50]:
            rel = full_path[len(prefix):]
            indent = "  " * min(rel.count("/"), depth)
            lines.append(f"{indent}{rel.rsplit('/', 1)[-1]}")
        return "\n".join(lines) if lines else "(empty)"

    def diff_summary(self, baseline: "VirtualFS") -> str:
        changed = [p for p, c in self.files.items() if baseline.files.get(p) != c]
        removed = [p for p in baseline.files if p not in self.files]
        if not changed and not removed:
            return "(no file changes)"
        parts = [f"M {path}" for path in sorted(changed)] + [f"D {path}" for path in sorted(removed)]
        return "\n".join(parts)


@dataclass
class TerminalProfile:
    """M1 profile: one tool (``bash``), text observations, virtual FS dynamics."""

    env_spec: EnvSpec
    root_files: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> VirtualFS:
        return VirtualFS(self.root_files)

    # ---- action parsing -------------------------------------------------

    def parse_action(self, action: Any) -> str:
        """Validate ``{"name": "bash", "args": {"command": str}}``; returns the command.

        Raises :class:`SynthEnvActionError` on enum/schema violations — the caller
        turns that into an observable rejection observation (violations channel),
        never a silent skip.
        """
        if not isinstance(action, dict):
            raise SynthEnvActionError(f"action must be an object, got {type(action).__name__}")
        name = action.get("name")
        if name != "bash":
            raise SynthEnvActionError(f"unknown action {name!r}; the terminal profile only supports 'bash'")
        args = action.get("args")
        if not isinstance(args, dict):
            raise SynthEnvActionError("action.args must be an object with a 'command' string")
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise SynthEnvActionError("action.args.command must be a non-empty string")
        if len(command) > self.env_spec.command_max_length:
            raise SynthEnvActionError(
                f"command exceeds the {self.env_spec.command_max_length}-character limit "
                f"({len(command)} characters)"
            )
        return command

    def match_danger(self, command: str) -> Optional[str]:
        """Return the blacklist pattern matched by the command, or None."""
        for pattern in self.env_spec.danger_patterns:
            if re.search(pattern, command):
                return pattern
        return None

    # ---- observations ---------------------------------------------------

    def initial_obs(self, goal: str) -> tuple[dict[str, str], str]:
        obs = {
            "channel": "text",
            "text": f"TASK: {goal}\n\nYou are in a synthetic shell. Use the bash tool one command at a time.",
        }
        return obs, f"cwd=/\n{self.snapshot().tree()}"

    def build_obs(self, outcome: ReplayOutcome, fs: VirtualFS, baseline: VirtualFS) -> tuple[dict[str, str], str]:
        text = outcome.stdout
        if outcome.stderr:
            text = f"{text}\n{outcome.stderr}".strip("\n") if text else outcome.stderr
        if outcome.exit_code:
            text = f"{text}\n[exit_code={outcome.exit_code}]".strip("\n")
        obs = {"channel": "text", "text": text or "(no output)"}
        return obs, f"{fs.diff_summary(baseline)}"

    def rejection_obs(self, message: str) -> dict[str, str]:
        return {"channel": "text", "text": f"synth_env: action rejected — {message}"}

    # ---- verify ---------------------------------------------------------

    def check_success(self, fs: VirtualFS, verifier_spec: Any) -> bool:
        kind = verifier_spec.kind
        if kind != "artifact_compare":
            # Fail loudly: an unimplemented verifier kind must never silently score 0.
            raise NotImplementedError(f"verifier kind {kind!r} is not implemented by the terminal profile")
        expected: dict[str, str] = verifier_spec.payload.get("expect_files", {})
        if not expected:
            raise ValueError("artifact_compare verifier requires a non-empty payload.expect_files")
        return all(fs.files.get(path) == content for path, content in expected.items())
