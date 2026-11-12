# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay dynamics engine (design §3.3/§3.4, M1): deterministic trie playback.

The dynamics of a terminal-profile synth environment are fully declared by the
asset package: every (cwd, command) pair that a task's golden path uses has a
pre-recorded ``ReplayOutcome``. There is zero model generation and zero real
command execution at runtime — the same input always produces the same
observation (HF discipline). A trie miss is a deterministic, observable error
(exit code 127), never a silent success.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


def normalize_command(command: str) -> str:
    """Collapse whitespace so semantically identical commands hash identically."""
    return " ".join(command.strip().split())


class ReplayOutcome(BaseModel):
    """The pre-recorded effect of one (cwd, command) pair."""

    model_config = {"extra": "forbid"}

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    # Deterministic filesystem effects declared by the package (never inferred).
    fs_writes: dict[str, str] = Field(default_factory=dict)
    fs_deletes: list[str] = Field(default_factory=list)
    cwd_to: Optional[str] = None
    done: bool = False
    reward_hint: Optional[float] = None


class ReplayMissError(LookupError):
    """Raised by lookup in strict mode; the engine converts it to an exit-127 outcome."""


class ReplayData:
    """Two-level trie ``{cwd -> normalized command -> outcome}`` built from package JSONL.

    Two levels are sufficient for terminal M1: the only semantic session state a
    replay lookup needs is the working directory. Splitting commands into a
    token-prefix trie adds nothing for whole-recorded pipelines like
    ``echo a > f`` and would break on shell syntax.
    """

    def __init__(self, entries: list[dict[str, Any]]):
        self._table: dict[str, dict[str, ReplayOutcome]] = {}
        for entry in entries:
            cwd = entry.get("cwd") or "/"
            command = entry.get("cmd")
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"replay entry missing 'cmd': {entry!r}")
            outcome = ReplayOutcome.model_validate({k: v for k, v in entry.items() if k not in ("cwd", "cmd")})
            self._table.setdefault(cwd, {})[normalize_command(command)] = outcome

    def __len__(self) -> int:
        return sum(len(commands) for commands in self._table.values())

    def lookup(self, cwd: str, command: str) -> ReplayOutcome:
        commands = self._table.get(cwd, {})
        outcome = commands.get(normalize_command(command))
        if outcome is None:
            raise ReplayMissError(f"no replay entry for cwd={cwd!r} cmd={normalize_command(command)!r}")
        return outcome


class ReplayEngine:
    """Applies a recorded outcome to a session's virtual filesystem — deterministically.

    The filesystem is passed in (duck-typed: needs ``apply(writes, deletes)`` and
    is mutated only through declared ``fs_writes``/``fs_deletes``); the engine
    never executes commands and never invents content.
    """

    MISS_EXIT_CODE = 127

    def __init__(self, replay: ReplayData):
        self._replay = replay

    def lookup(self, cwd: str, command: str) -> ReplayOutcome:
        try:
            return self._replay.lookup(cwd, command)
        except ReplayMissError:
            return self._miss_outcome(command)

    def _miss_outcome(self, command: str) -> ReplayOutcome:
        # Deterministic and observable: the command "ran" and failed with 127.
        return ReplayOutcome(
            stderr=f"bash: {command.strip().split()[0] if command.strip() else command}: command not found\n"
            "synth_env: no replay entry for this command (replay dynamics)\n",
            exit_code=self.MISS_EXIT_CODE,
        )

    def execute(self, fs: Any, cwd: str, command: str) -> tuple[ReplayOutcome, str]:
        """Look up and apply the outcome; returns ``(outcome, new_cwd)``."""
        outcome = self.lookup(cwd, command)
        if outcome.fs_writes or outcome.fs_deletes:
            fs.apply(dict(outcome.fs_writes), list(outcome.fs_deletes))
        return outcome, outcome.cwd_to or cwd
