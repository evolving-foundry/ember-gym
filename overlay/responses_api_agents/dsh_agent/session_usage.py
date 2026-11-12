# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-usage extraction from dsh session journals (no nemo_gym dependencies).

dsh persists a zstd-compressed JSONL journal per session under
`<DSH_HOME|~/.dsh>/sessions/<cwd-slug>/session-*/session.v*.jsonl.zstd`. The
leading record is `{"type": "session", "createdAt": <epoch-ms>, ...}`; each
model turn appends an `assistant/message` record carrying
`data.usage = {inputTokens, outputTokens, totalTokens, cacheReadTokens}`.

dsh speaks the anthropic-messages protocol, where cache reads are reported
OUTSIDE inputTokens (`totalTokens == inputTokens + outputTokens +
cacheReadTokens`). Parsed results are re-expressed in OpenAI semantics:
input_tokens includes cached tokens and total_tokens = input + output.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional


LOG = logging.getLogger(__name__)


def _dsh_posix_path(cwd: Path) -> str:
    raw = os.fspath(cwd)
    posix_rooted = raw.startswith("/") or (raw.startswith("\\") and not raw.startswith("\\\\"))
    if os.name == "nt" and posix_rooted:
        # A Windows-native Path('/mnt/c/...') resolves to <current-drive>:\mnt\c.
        # dsh journals observed from WSL use the original POSIX identity.
        return raw.replace("\\", "/").rstrip("/")
    return Path(cwd).resolve().as_posix().rstrip("/")


def dsh_session_slug(cwd: Path) -> str:
    """dsh keys its session journal directories by the (slugified) process cwd.

    Derived from observed journals (e.g. cwd /mnt/c/WorkSpace/Gym →
    `--mnt-c-WorkSpace-Gym--`): the posix path is treated as if it carried a
    trailing separator, then wrapped in dashes on both sides.
    """
    posix = _dsh_posix_path(cwd)
    return "-" + (posix + "/").replace("/", "-") + "-"


def _dsh_session_slug_fallbacks(cwd: Path) -> list:
    """Alternate slug spellings, tried when the primary directory is absent."""
    posix = _dsh_posix_path(cwd)
    return ["-" + posix.replace("/", "-") + "-"]


def _session_created_at(session_path: Path) -> Optional[float]:
    """Epoch seconds from the journal's leading `session` record (createdAt is epoch ms)."""
    try:
        import zstandard

        with Path(session_path).open("rb") as handle:
            head = zstandard.ZstdDecompressor().stream_reader(handle).read(65536)
        record = json.loads(head.split(b"\n", 1)[0].decode("utf-8", errors="replace"))
    except Exception:
        return None
    if record.get("type") != "session":
        return None
    created = record.get("createdAt")
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        return float(created) / 1000.0
    return None


def parse_dsh_session_usage(session_path: Path) -> Optional[dict]:
    """Sum `usage` across assistant/message records of one dsh session journal.

    Returns None when nothing usable is found (missing/corrupt/usage-free journal).
    """
    try:
        import zstandard

        with Path(session_path).open("rb") as handle:
            text = zstandard.ZstdDecompressor().stream_reader(handle).read().decode("utf-8", errors="replace")
    except Exception:
        return None

    input_tokens = output_tokens = cached_tokens = 0
    saw_usage = False
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a partially flushed trailing record
        if record.get("type") != "assistant/message":
            continue
        usage = (record.get("data") or {}).get("usage") or {}
        if not isinstance(usage, dict) or not usage:
            continue
        saw_usage = True
        input_tokens += int(usage.get("inputTokens") or 0)
        output_tokens += int(usage.get("outputTokens") or 0)
        cached_tokens += int(usage.get("cacheReadTokens") or 0)
    if not saw_usage:
        return None

    input_tokens += cached_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_tokens": cached_tokens,
    }


def collect_dsh_session_usage(sessions_root: Path, cwd: Path, started_at: float) -> Optional[dict]:
    """Parse the newest dsh session journal for `cwd` whose session opened at/after `started_at`.

    `started_at` guards against picking up a stale session left in the same slug
    by an earlier run; the small slack absorbs clock jitter between dsh and us.
    """
    journal_roots = [Path(sessions_root) / dsh_session_slug(cwd)] + [
        Path(sessions_root) / fallback for fallback in _dsh_session_slug_fallbacks(cwd)
    ]
    best_path = None
    best_created = -1.0
    for journal_root in journal_roots:
        try:
            candidates = list(journal_root.glob("session-*/session.v*.jsonl.zstd"))
        except OSError:
            continue
        for path in candidates:
            created = _session_created_at(path)
            if created is not None and created >= started_at - 5.0 and created > best_created:
                best_path = path
                best_created = created
        if best_path is not None:
            break
    if best_path is None:
        return None
    return parse_dsh_session_usage(best_path)
