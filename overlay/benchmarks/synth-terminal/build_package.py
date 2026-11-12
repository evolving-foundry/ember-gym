# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the M1/M3 demo asset packages (Gym-authored demos, hand-placed on purpose).

The ADF-side publisher is live: AgentGym-DataDesigner
``scripts/gym_publish_env.py`` materializes ADF-frozen env packages into this
overlay (assets zip + config/manifest/prepare shell + prompts + overlay
registration). These demo packages stay Gym-authored — they exercise host
dynamics directly and are not ADF synthesis products.

Produces two deterministic terminal-profile packages over the same ``ref_id``
(demo of multi-version coexistence, design §3.13):

- ``assets/synth_terminal_demo_v1.zip`` — replay dynamics (M1): three tasks
  covering the win path, step truncation, and the dangerous-command blacklist;
- ``assets/synth_terminal_demo_v2_hybrid.zip`` — hybrid dynamics (M3 类真实,
  design §3.4): the same skeleton with ``dynamics: "hybrid"`` and a safe-subset
  ``real_exec_allowlist``; allowlisted commands really execute against the
  materialized seed FS, everything else falls back to replay.

Package fingerprints are content-addressed (sha256 over per-file content
hashes, design §3.5), recomputed here and frozen into both the manifest and
every task row. Re-run after editing package content:

    uv run python overlay/benchmarks/synth-terminal/build_package.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path


# Allow running as a script from anywhere: the overlay root must be importable
# (overlay/resources_servers/... is a namespace package sibling of the wheel's).
OVERLAY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OVERLAY_ROOT))

from resources_servers.synth_env.package_loader import compute_fingerprint  # noqa: E402


REF_ID = "synth-terminal-demo"
VERSION = "1"
PROFILE = "terminal"

MANIFEST_NAME = "manifest.json"

README_TEXT = "# Demo Service\n\nversion: 1.1.0\n\nA tiny synthetic service used by the M1 demo tasks.\n"
VERSION_TXT = "1.2.0\n"

DANGER_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"mkfs(\.\w+)?\s",
    r"dd\s+if=",
    r":\(\)\{.*\};\};",  # fork bomb
    r"shutdown|reboot|halt",
    r"chmod\s+-R\s+777\s+/",
]

ENV_SPEC = {
    "action_schema": {
        "name": "bash",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "maxLength": 2000}},
            "required": ["command"],
        },
    },
    "command_max_length": 2000,
    "danger_patterns": DANGER_PATTERNS,
    "state_schema": {},
}

# M3 hybrid variant: same skeleton, safe-subset real execution on top of replay.
# The allowlist covers only read-only-ish coreutils the demo tasks need; anything
# else (and any shell syntax) takes the replay fallback.
ENV_SPEC_V2_HYBRID = dict(
    ENV_SPEC,
    dynamics="hybrid",
    real_exec_allowlist=["cat", "ls", "cp", "grep", "wc", "df"],
    exec_timeout_s=5.0,
    exec_max_output_chars=65536,
    hybrid_miss_policy="replay",
)

FS_SNAPSHOT = [
    {"path": "/home/user/README.md", "content": README_TEXT},
    {"path": "/home/user/notes/draft.md", "content": "- draft note\n"},
]

REPLAY_ENTRIES = [
    # Shared discovery commands (cwd is always "/" in M1; commands use absolute paths).
    {"cwd": "/", "cmd": "ls /home/user", "stdout": "README.md\nnotes\n"},
    {"cwd": "/", "cmd": "cat /home/user/README.md", "stdout": README_TEXT},
    # synth-term-0001 win path: read the version, write the new one.
    {
        "cwd": "/",
        "cmd": f"echo {VERSION_TXT.strip()} > /home/user/version.txt",
        "stdout": "",
        "fs_writes": {"/home/user/version.txt": VERSION_TXT},
        "done": True,
    },
    # synth-term-0002: the only recorded command; its goal needs unrecorded ones.
    {"cwd": "/", "cmd": "ls /var/log", "stdout": "syslog\n"},
    # synth-term-0003: harmless discovery before the (blocked) dangerous command.
    {"cwd": "/", "cmd": "df -h", "stdout": "Filesystem  Size  Used  Use%  Mounted on\n/dev/sda1  40G  21G  53%  /\n"},
]


def _task(task_id: str, goal: str, expect_files: dict[str, str], max_steps: int, fingerprint: str,
          version: str = VERSION) -> dict:
    return {
        "task_id": task_id,
        "env_release": {"ref_id": REF_ID, "version": version, "fingerprint": fingerprint},
        "goal": goal,
        "success_check": {"verifier_spec": {"kind": "artifact_compare", "payload": {"expect_files": expect_files}}},
        "max_steps": max_steps,
        "seed": 42,
    }


def build_files(env_spec: dict = None) -> dict[str, str]:
    """Package file contents covered by the fingerprint.

    ``tasks.jsonl`` and ``manifest.json`` are excluded: task rows embed the
    fingerprint itself, so the package identity covers the contract + dynamics
    + seed state (env_spec / replay / FS snapshot). Task integrity is enforced
    structurally at load time (every row must pin the package's own
    ref_id/version/fingerprint).
    """
    return {
        "env_spec.json": json.dumps(env_spec if env_spec is not None else ENV_SPEC, indent=2, ensure_ascii=False) + "\n",
        "replay/replay.jsonl": "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in REPLAY_ENTRIES),
        "fs/snapshot.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in FS_SNAPSHOT),
    }


def build_tasks(version: str, fingerprint: str) -> list[dict]:
    if version == VERSION:
        return [
            _task(
                "synth-term-0001",
                "Read the service version from /home/user/README.md, then write the upgraded "
                "version '1.2.0' to /home/user/version.txt.",
                {"/home/user/version.txt": VERSION_TXT},
                max_steps=8,
                fingerprint=fingerprint,
            ),
            _task(
                "synth-term-0002",
                "Archive /var/log into /home/user/logs.tar.gz (no replay entry exists for this — "
                "the session is expected to run out of steps).",
                {"/home/user/logs.tar.gz": "irrelevant"},
                max_steps=2,
                fingerprint=fingerprint,
            ),
            _task(
                "synth-term-0003",
                "Free disk space on this host (the recorded golden path attempts rm -rf /, "
                "which the environment must block).",
                {"/home/user/done.txt": "ok\n"},
                max_steps=5,
                fingerprint=fingerprint,
            ),
        ]
    # v2 hybrid: the allowlisted subset really executes; everything else (and any
    # shell syntax) falls back to replay — same discipline, 类真实 dynamics.
    return [
        _task(
            "synth-term-h1",
            "Back up the service README: copy /home/user/README.md to /home/user/backup.md "
            "(cp is on the real-exec allowlist — the artifact must really exist).",
            {"/home/user/backup.md": README_TEXT},
            max_steps=4,
            fingerprint=fingerprint,
            version=version,
        ),
        _task(
            "synth-term-h2",
            "Archive /var/log into /home/user/logs.tar.gz (tar is not allowlisted and shell "
            "syntax never reaches a subprocess — the session is expected to run out of steps).",
            {"/home/user/logs.tar.gz": "irrelevant"},
            max_steps=2,
            fingerprint=fingerprint,
            version=version,
        ),
        _task(
            "synth-term-h3",
            "Free disk space on this host (the recorded golden path attempts rm -rf /, "
            "which the environment must block before any dynamics run).",
            {"/home/user/done.txt": "ok\n"},
            max_steps=5,
            fingerprint=fingerprint,
            version=version,
        ),
    ]


def build_package(version: str, env_spec: dict, out_name: str, description: str) -> Path:
    files = build_files(env_spec)
    fingerprint = compute_fingerprint(REF_ID, version, PROFILE, files)
    files["tasks.jsonl"] = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in build_tasks(version, fingerprint))

    manifest = {
        "ref_id": REF_ID,
        "version": version,
        "fingerprint": fingerprint,
        "profile": PROFILE,
        "description": description,
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Fixed entry metadata: writestr with a plain name stamps the wall clock
        # into each ZipInfo.date_time, which would break byte reproducibility.
        fixed_info = zipfile.ZipInfo(MANIFEST_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        fixed_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(fixed_info, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)

    out_path = Path(__file__).resolve().parent / "assets" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buffer.getvalue())
    print(f"wrote {out_path} (fingerprint {fingerprint})")
    return out_path


def main() -> None:
    build_package(VERSION, ENV_SPEC, "synth_terminal_demo_v1.zip",
                  "M1 demo terminal package: win path, step truncation, dangerous-command blacklist.")
    build_package("2", ENV_SPEC_V2_HYBRID, "synth_terminal_demo_v2_hybrid.zip",
                  "M3 demo terminal package (hybrid 类真实): safe-subset real execution over replay fallback.")


if __name__ == "__main__":
    main()
