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
"""Rubric judgment adapter (design doc §3.3 / §5.3 scenario C): when a benchmark has no programmatic
verifier, candidate rollouts are scored against a *frozen* rubric library snapshot instead.

Flow: load + integrity-check the frozen snapshot → group rollout rows per query (n =
``rubric_group_size``) → score complete groups through a judge backend → aggregate per-rollout
weighted rewards → emit per-criterion trusted/overfit signals (P7) that merge into
``OverfitGuard.criterion_signals``.

Discipline:
- Judge endpoint failure is an explicit failure (:class:`RubricJudgeError`) — never an all-zero
  degradation (§7.3(5));
- The one-way Gym → data_designer dependency (§3.1) lives behind :class:`DataDesignerRubricBackend`
  (lazy import); tests inject fake backends instead, so Gym core tests never need the extra;
- The snapshot loader mirrors DataDesigner ``evolve.snapshot.load_frozen_snapshot`` stdlib-only,
  recomputing both embedded digests with the exact same canonical JSON serialization — a tampered
  snapshot is refused before any rollout runs (§7.2).
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from eval_service.profile import BenchmarkSpec


SNAPSHOT_SCHEMA = "rubric-library-snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
EVIDENCE_SCOPE_FROZEN_SEED_ONLY = "frozen_seed_only"
MANIFEST_DIGEST_PREFIX = "sha256:"

# Rollout rows emitted by the harness must carry the query and the candidate's final answer so
# they can be grouped and judged (the rubric benchmark's config is responsible for emitting them).
QUERY_FIELD = "query"
ANSWER_FIELD = "final_answer"


class RubricLibraryError(RuntimeError):
    """The frozen rubric library snapshot is missing, tampered with, or invalid (§7.2: refuse to start)."""


class RubricJudgeError(RuntimeError):
    """The judge backend failed or violated the binary score contract (explicit failure, no all-zero fallback)."""


def canonical_json(value: Any) -> str:
    # Byte-compatible with data_designer.contracts.provenance.canonical_json — the snapshot's
    # embedded digests are recomputed locally with this exact serialization.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(value: Any) -> str:
    return MANIFEST_DIGEST_PREFIX + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RubricCriterion:
    rubric_id: str
    title: str
    description: str
    weight: float


@dataclass(frozen=True)
class RubricVerdict:
    rewards: tuple[float, ...]
    rubric_library_sha256: str
    rubric_ids: tuple[str, ...]
    rubric_weights: tuple[float, ...]
    # Per-criterion judge score==1 hits across all scored rollouts (§6.1 group_pass_counts).
    group_pass_counts: tuple[int, ...]
    # rubric_id -> {hit_count, scored_rollouts, trusted, reason} (P7 OverfitGuard merge).
    criterion_signals: dict[str, dict[str, Any]]
    judge_model: str
    scored_rollouts: int
    ungrouped_rollouts: int
    evidence_rows: tuple[dict[str, Any], ...]


def load_rubric_library(fpath: str | Path) -> tuple[tuple[RubricCriterion, ...], str]:
    """Parse + integrity-check a frozen snapshot; returns (criteria, rubric_library_sha256).

    Raises :class:`RubricLibraryError` on any schema/integrity failure.
    """
    path = Path(fpath)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RubricLibraryError(f"rubric library snapshot not found: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RubricLibraryError(f"rubric library snapshot unreadable: {path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("snapshot_schema") != SNAPSHOT_SCHEMA
        or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
    ):
        raise RubricLibraryError(f"unsupported rubric snapshot schema: {path}")
    if payload.get("evidence_scope") != EVIDENCE_SCOPE_FROZEN_SEED_ONLY:
        raise RubricLibraryError(f"rubric snapshot must be exported with evidence_scope=frozen_seed_only: {path}")
    annotated = str(payload.get("snapshot_sha256") or "")
    body = json.dumps(
        {key: value for key, value in payload.items() if key != "snapshot_sha256"},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if annotated != hashlib.sha256(body.encode("utf-8")).hexdigest():
        raise RubricLibraryError(f"rubric snapshot integrity check failed: {path}")
    library = payload.get("rubric_library")
    if not isinstance(library, list) or not library:
        raise RubricLibraryError(f"snapshot rubric_library must be a non-empty list: {path}")
    library_digest = str(payload.get("rubric_library_sha256") or "")
    if library_digest != content_digest([dict(item) for item in library]):
        raise RubricLibraryError(f"snapshot rubric library digest mismatch: {path}")
    return _validate_library(library), library_digest


def _validate_library(library: list[dict[str, Any]]) -> tuple[RubricCriterion, ...]:
    """Mirror of the DataDesigner rubric protocol's validate_rubrics (stdlib-only)."""
    criteria: list[RubricCriterion] = []
    seen_ids: set[str] = set()
    has_positive_weight = False
    for index, rubric in enumerate(library):
        if not isinstance(rubric, dict):
            raise RubricLibraryError(f"rubric {index} must be a mapping")
        rubric_id = str(rubric.get("rubric_key") or "").strip()
        title = str(rubric.get("title") or "").strip()
        description = str(rubric.get("description") or "").strip()
        if not rubric_id:
            raise RubricLibraryError(f"rubric {index} rubric_key must be a non-empty string")
        if not title:
            raise RubricLibraryError(f"rubric {index} title must be a non-empty string")
        if not description:
            raise RubricLibraryError(f"rubric {index} description must be a non-empty string")
        weight = rubric.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RubricLibraryError(f"rubric {index} weight must be numeric")
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight == 0.0:
            raise RubricLibraryError(f"rubric {index} weight must be finite and non-zero")
        if rubric_id in seen_ids:
            raise RubricLibraryError(f"rubric keys must be unique: duplicate {rubric_id!r}")
        seen_ids.add(rubric_id)
        has_positive_weight |= numeric_weight > 0.0
        criteria.append(RubricCriterion(rubric_id, title, description, numeric_weight))
    if not has_positive_weight:
        raise RubricLibraryError("rubric library requires at least one positive-weight criterion")
    return tuple(criteria)


class RubricJudgeBackend(Protocol):
    """Judge seam: production wires DataDesigner's rubric judge transport; tests script fakes."""

    def score_rollout(self, *, query: str, final_answer: str, descriptions: tuple[str, ...]) -> tuple[int, ...]:
        """Return one binary score per criterion (aligned with ``descriptions``)."""
        ...

    def aggregate(self, scores: tuple[int, ...], weights: tuple[float, ...]) -> float:
        """Weighted binary aggregation with positive-weight denominator, value range [0, 1]-ish
        (negative-weight criteria make violation penalties pull below 0)."""
        ...


class DataDesignerRubricBackend:
    """Production backend: the one-way Gym → data_designer.rubric dependency (§3.1).

    Endpoint credentials flow through the HEALTHBENCH_JUDGE_* env channel (§6.4); the judge model
    name comes from the profile's ``rubric_judge`` pin.
    """

    def __init__(self, model: str) -> None:
        try:
            from data_designer.rubric import aggregate_weighted_reward, configure_judge_endpoints
        except ImportError as error:
            raise RubricJudgeError(
                "data_designer.rubric is not importable — install the rubric extra (uv sync --extra rubric); "
                "rubric-scored benchmarks fail explicitly instead of degrading (§7.3(5))"
            ) from error
        self._transport = configure_judge_endpoints(
            api_base=os.environ.get("HEALTHBENCH_JUDGE_API_BASE", ""),
            api_key=os.environ.get("HEALTHBENCH_JUDGE_API_KEY", ""),
            model=model,
        )
        self._aggregate = aggregate_weighted_reward

    def score_rollout(self, *, query: str, final_answer: str, descriptions: tuple[str, ...]) -> tuple[int, ...]:
        try:
            result = self._transport.call_judge(query=query, final_answer=final_answer, descriptions=descriptions)
        except Exception as error:
            # Translate the DataDesigner failure vocabulary (JudgeGiveUpError on attempt/time
            # budget exhaustion, protocol/network errors) into the Gym seam's explicit-failure
            # type so the service converts it into a structured RUBRIC_JUDGE_FAILED result.
            raise RubricJudgeError(f"data_designer judge transport failed: {error}") from error
        return tuple(result.scores)

    def aggregate(self, scores: tuple[int, ...], weights: tuple[float, ...]) -> float:
        return float(self._aggregate(list(scores), list(weights)))


def judge_rubric_rollouts(
    rows: Sequence[dict[str, Any]],
    criteria: Sequence[RubricCriterion],
    *,
    backend: RubricJudgeBackend,
    group_size: int,
    library_sha256: str,
    judge_model: str,
) -> RubricVerdict:
    """Score rollout rows against the frozen library; returns the aggregate verdict.

    Rows are grouped per query; only complete n=``group_size`` groups are scored (RuACT protocol:
    a complete group is required). Incomplete leftovers are excluded explicitly — counted in
    ``ungrouped_rollouts`` and surfaced as a result warning, never silently averaged in. When no
    complete group exists the judgment fails explicitly.
    """
    if not criteria:
        raise RubricJudgeError("rubric criteria must be non-empty")
    groups: dict[str, list[tuple[int, str]]] = {}
    for index, row in enumerate(rows):
        query = row.get(QUERY_FIELD)
        answer = row.get(ANSWER_FIELD)
        if not isinstance(query, str) or not query.strip() or not isinstance(answer, str):
            raise RubricJudgeError(
                f"rollout row {index} must carry non-empty {QUERY_FIELD!r}/{ANSWER_FIELD!r} fields for rubric scoring"
            )
        groups.setdefault(query, []).append((index, answer))
    complete = {query: members for query, members in groups.items() if len(members) == group_size}
    ungrouped_rollouts = sum(len(members) for members in groups.values() if len(members) != group_size)
    if not complete:
        summary = ", ".join(f"{query!r}={len(members)}" for query, members in sorted(groups.items())[:5])
        raise RubricJudgeError(
            f"no complete rollout group of {group_size} for rubric scoring (per-query counts: {summary})"
        )

    descriptions = tuple(criterion.description for criterion in criteria)
    weights = tuple(criterion.weight for criterion in criteria)
    rubric_ids = tuple(criterion.rubric_id for criterion in criteria)
    rewards: list[float] = []
    hits = [0] * len(criteria)
    evidence_rows: list[dict[str, Any]] = []
    for query in sorted(complete):
        for rollout_index, (_, answer) in enumerate(sorted(complete[query])):
            scores = backend.score_rollout(query=query, final_answer=answer, descriptions=descriptions)
            if len(scores) != len(criteria) or any(score not in (0, 1) for score in scores):
                raise RubricJudgeError("judge backend returned a misaligned or non-binary score vector")
            reward = backend.aggregate(scores, weights)
            rewards.append(reward)
            for position, score in enumerate(scores):
                hits[position] += score
            evidence_rows.append(
                {
                    "query": query,
                    "rollout_index": rollout_index,
                    "rubric_ids": list(rubric_ids),
                    "rubric_scores": list(scores),
                    "reward": reward,
                }
            )

    scored_rollouts = len(rewards)
    criterion_signals: dict[str, dict[str, Any]] = {}
    for criterion, hit_count in zip(criteria, hits):
        trusted = 0 < hit_count < scored_rollouts
        criterion_signals[criterion.rubric_id] = {
            "hit_count": hit_count,
            "scored_rollouts": scored_rollouts,
            "trusted": trusted,
            "reason": "" if trusted else "constant_score_column",
        }
    return RubricVerdict(
        rewards=tuple(rewards),
        rubric_library_sha256=library_sha256,
        rubric_ids=rubric_ids,
        rubric_weights=weights,
        group_pass_counts=tuple(hits),
        criterion_signals=criterion_signals,
        judge_model=judge_model,
        scored_rollouts=scored_rollouts,
        ungrouped_rollouts=ungrouped_rollouts,
        evidence_rows=tuple(evidence_rows),
    )


def rubric_library_path(benchmark: BenchmarkSpec, layout) -> Path:
    """Resolve the benchmark's frozen snapshot path (absolute paths pass through)."""
    fpath = Path(benchmark.rubric_library_fpath or "")
    if fpath.is_absolute():
        return fpath
    return layout.resolve_data(benchmark.rubric_library_fpath, must_exist=False)


__all__ = [
    "ANSWER_FIELD",
    "QUERY_FIELD",
    "DataDesignerRubricBackend",
    "RubricCriterion",
    "RubricJudgeBackend",
    "RubricJudgeError",
    "RubricLibraryError",
    "RubricVerdict",
    "content_digest",
    "judge_rubric_rollouts",
    "load_rubric_library",
    "rubric_library_path",
]
