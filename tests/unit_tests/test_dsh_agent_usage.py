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
import json
from pathlib import Path

import zstandard
from responses_api_agents.dsh_agent.session_usage import (
    collect_dsh_session_usage,
    dsh_session_slug,
    parse_dsh_session_usage,
)


JOURNAL_NAME = "session.v2.jsonl.zstd"


def _compress(lines: list) -> bytes:
    return zstandard.ZstdCompressor().compress(("\n".join(lines) + "\n").encode())


def _write_journal(tmp_path: Path, slug: str, session_id: str, created_ms: int, records: list) -> Path:
    session_dir = tmp_path / "sessions" / slug / f"session-{session_id}"
    session_dir.mkdir(parents=True)
    lines = [json.dumps({"type": "session", "createdAt": created_ms})]
    lines.extend(json.dumps(record) for record in records)
    path = session_dir / JOURNAL_NAME
    path.write_bytes(_compress(lines))
    return path


def _assistant_record(input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> dict:
    return {
        "type": "assistant/message",
        "data": {
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cacheReadTokens": cache_read_tokens,
                "totalTokens": input_tokens + output_tokens + cache_read_tokens,
            }
        },
    }


def test_session_slug_matches_observed_dsh_journals():
    # Format observed in ~/.dsh/sessions for real dsh runs.
    assert dsh_session_slug(Path("/mnt/c/WorkSpace/Gym")) == "--mnt-c-WorkSpace-Gym--"
    assert dsh_session_slug(Path("/home/chad/.dsh")) == "--home-chad-.dsh--"


def test_parse_sums_usage_in_openai_semantics(tmp_path):
    path = tmp_path / JOURNAL_NAME
    path.write_bytes(
        _compress(
            [
                json.dumps({"type": "session", "createdAt": 1700000000000}),
                json.dumps(_assistant_record(100, 5, cache_read_tokens=10)),
                json.dumps(_assistant_record(200, 7)),
                # noise: usage-free records must be ignored
                json.dumps({"type": "user/message", "data": {}}),
            ]
        )
    )
    usage = parse_dsh_session_usage(path)
    # cache reads arrive OUTSIDE inputTokens in dsh, so input = 100+200+10.
    assert usage == {"input_tokens": 310, "output_tokens": 12, "total_tokens": 322, "cached_tokens": 10}


def test_parse_returns_none_on_corrupt_or_usage_free_journals(tmp_path):
    corrupt = tmp_path / "corrupt.zstd"
    corrupt.write_bytes(b"not-zstd")
    assert parse_dsh_session_usage(corrupt) is None

    empty = tmp_path / "empty.zstd"
    empty.write_bytes(_compress([json.dumps({"type": "session", "createdAt": 1700000000000})]))
    assert parse_dsh_session_usage(empty) is None


def test_collect_picks_newest_session_at_or_after_start(tmp_path):
    cwd = Path("/mnt/c/WorkSpace/Gym")
    slug = dsh_session_slug(cwd)
    _write_journal(tmp_path, slug, "old", 1000, [_assistant_record(1, 1)])
    _write_journal(tmp_path, slug, "new", 2000, [_assistant_record(300, 9)])

    usage = collect_dsh_session_usage(tmp_path / "sessions", cwd, started_at=1.5)
    assert usage == {"input_tokens": 300, "output_tokens": 9, "total_tokens": 309, "cached_tokens": 0}

    # both sessions predate the start -> nothing is attributed
    assert collect_dsh_session_usage(tmp_path / "sessions", cwd, started_at=9999.0) is None
    # unknown cwd -> None
    assert collect_dsh_session_usage(tmp_path / "sessions", Path("/elsewhere"), started_at=0.0) is None
