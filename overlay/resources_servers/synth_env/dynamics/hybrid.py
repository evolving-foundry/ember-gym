# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid dynamics engine (design §3.4, M3 类真实): safe-subset real execution + replay fallback.

The package's ``env_spec`` declares the contract:

- ``dynamics: "hybrid"`` — enables this engine (default ``"replay"`` keeps M1 semantics);
- ``real_exec_allowlist`` — command *heads* (first token) that really execute;
- ``exec_timeout_s`` / ``exec_max_output_chars`` — subprocess guards;
- ``hybrid_miss_policy`` — what happens to commands outside the safe subset:
  ``"replay"`` (default, the §3.4 "其余模拟" semantics) or ``"reject"``.

Safety shape of "受控子进程": the session's virtual FS is materialized into a
fresh per-step private temp directory and **destroyed afterwards** (no session-
long state, nothing survives a step); execution is ``shell=False`` over
``shlex.split`` with a minimal environment (PATH/LANG/HOME only, HOME pinned to
the temp dir) — no host environment, no network surface added by the engine. Any
shell metacharacter in the command (redirection, pipes, substitution) marks it
outside the safe subset — those commands take the miss policy, they are never
handed to a shell. The dangerous-command blacklist still runs at parse time,
*before* the engine is reached. Because the virtual FS is addressed by absolute
paths, argv tokens starting with ``/`` are remapped into the temp tree (and the
temp prefix is rewritten out of captured output), so commands observe the same
paths the virtual FS reports.

Determinism note (§3.4 "类真实"): real execution produces real output — the
engine guarantees the *frame* is deterministic (same FS seed, same command →
same materialized tree, same guards) but does not promise byte-identical stdout
for commands whose output embeds the world (timestamps, ordering). Verify still
runs on the FS diff, which is content-addressed.

Real-execution results are scanned back into the virtual FS (writes/deletes vs.
the pre-step state) so observations and ``artifact_compare`` verify see one
coherent filesystem.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from resources_servers.synth_env.dynamics.replay import ReplayEngine, ReplayOutcome

# Shell metacharacters: any occurrence means the command needs a shell to mean
# what it says — outside the safe subset by definition (never shell=True).
_SHELL_METACHARS = set(">|&;`$(){}[]*?~!\n\\")


class HybridEngine:
    """Allowlisted commands really execute; everything else takes the miss policy."""

    REJECT_EXIT_CODE = 126
    TIMEOUT_EXIT_CODE = 124
    TRUNCATION_MARKER = "\n[output truncated]"

    def __init__(
        self,
        *,
        replay: ReplayEngine,
        allowlist: list[str],
        timeout_s: float = 5.0,
        max_output_chars: int = 65536,
        miss_policy: str = "replay",
    ):
        if miss_policy not in ("replay", "reject"):
            raise ValueError(f"hybrid_miss_policy must be 'replay' or 'reject', got {miss_policy!r}")
        self._replay = replay
        self._allowlist = set(allowlist)
        self._timeout_s = timeout_s
        self._max_output_chars = max_output_chars
        self._miss_policy = miss_policy

    def execute(self, fs: Any, cwd: str, command: str) -> tuple[ReplayOutcome, str]:
        """Run one command; same contract as :meth:`ReplayEngine.execute`."""
        if self._is_real_exec_eligible(command):
            return self._execute_real(fs, cwd, command)
        if self._miss_policy == "reject":
            return (
                ReplayOutcome(
                    stderr="synth_env: command is outside the real-exec allowlist "
                    f"(hybrid_miss_policy=reject): {command.strip()}\n",
                    exit_code=self.REJECT_EXIT_CODE,
                ),
                cwd,
            )
        return self._replay.execute(fs, cwd, command)

    def _is_real_exec_eligible(self, command: str) -> bool:
        if not self._allowlist:
            return False
        if any(char in _SHELL_METACHARS for char in command):
            return False
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        return bool(tokens) and tokens[0] in self._allowlist

    def _execute_real(self, fs: Any, cwd: str, command: str) -> tuple[ReplayOutcome, str]:
        before = dict(fs.files)
        with tempfile.TemporaryDirectory(prefix="synth_env_hybrid_") as tmp:
            root = Path(tmp)
            self._materialize(root, before)
            workdir = root / cwd.lstrip("/") if cwd.strip("/") else root
            if not workdir.is_dir():
                workdir = root
            # Virtual FS paths are absolute ("/home/user/..."); inside the sandbox
            # they must resolve against the temp tree, so rewrite argv tokens, and
            # rewrite the temp prefix back out of captured output for observation.
            tokens = [f"{root}{token}" if token.startswith("/") else token for token in shlex.split(command)]
            try:
                completed = subprocess.run(
                    tokens,
                    cwd=str(workdir),
                    capture_output=True,
                    timeout=self._timeout_s,
                    shell=False,
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "HOME": str(root)},
                    check=False,
                )
                stdout = completed.stdout.decode("utf-8", errors="replace").replace(tmp, "")
                stderr = completed.stderr.decode("utf-8", errors="replace").replace(tmp, "")
                exit_code = completed.returncode
            except subprocess.TimeoutExpired as error:
                stdout = (error.stdout or b"").decode("utf-8", errors="replace").replace(tmp, "")
                stderr = f"synth_env: real execution timed out after {self._timeout_s}s\n"
                exit_code = self.TIMEOUT_EXIT_CODE
            stdout = self._truncate(stdout)
            stderr = self._truncate(stderr)
            self._scan_back(root, fs, before)
        return ReplayOutcome(stdout=stdout, stderr=stderr, exit_code=exit_code), cwd

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        return text[: self._max_output_chars] + self.TRUNCATION_MARKER

    @staticmethod
    def _materialize(root: Path, files: dict[str, str]) -> None:
        for path, content in files.items():
            target = root / path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _scan_back(self, root: Path, fs: Any, before: dict[str, str]) -> None:
        after = {
            f"/{relative.relative_to(root).as_posix()}": relative.read_text(encoding="utf-8", errors="replace")
            for relative in sorted(root.rglob("*"))
            if relative.is_file()
        }
        writes = {path: content for path, content in after.items() if before.get(path) != content}
        deletes = [path for path in before if path not in after]
        if writes or deletes:
            fs.apply(writes, deletes)


def build_dynamics_engine(env_spec: Any, replay: ReplayEngine) -> Any:
    """Dynamics selector (§3.4): the package's env_spec picks replay or hybrid."""
    if getattr(env_spec, "dynamics", "replay") == "hybrid":
        return HybridEngine(
            replay=replay,
            allowlist=list(env_spec.real_exec_allowlist),
            timeout_s=float(env_spec.exec_timeout_s),
            max_output_chars=int(env_spec.exec_max_output_chars),
            miss_policy=str(env_spec.hybrid_miss_policy),
        )
    return replay


__all__ = ["HybridEngine", "build_dynamics_engine"]
