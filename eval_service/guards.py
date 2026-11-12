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
"""Persistent guard ledger: overfit detection (OF-3/OF-4), lineage cost accounting (EC-4), and
manifest-digest history for digest_changed warnings (HF-4)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


TREND_WINDOW = 5


def score_trend_summary(history: list[Optional[float]]) -> str:
    """Summarize the last TREND_WINDOW reward means, e.g. "monotonic_up_5" (OF-3)."""
    scores = [s for s in history if s is not None][-TREND_WINDOW:]
    if len(scores) < 2:
        return "insufficient_history"
    if all(b > a for a, b in zip(scores, scores[1:])):
        return f"monotonic_up_{len(scores)}"
    if all(b < a for a, b in zip(scores, scores[1:])):
        return f"monotonic_down_{len(scores)}"
    if all(b == a for a, b in zip(scores, scores[1:])):
        return f"flat_{len(scores)}"
    return "mixed"


class Ledger:
    """JSON-file-backed record of every evaluation the service has executed."""

    def __init__(self, fpath: str | Path) -> None:
        self.fpath = Path(fpath)
        self._evals: list[dict[str, Any]] = []
        if self.fpath.is_file():
            with open(self.fpath, encoding="utf-8") as fh:
                self._evals = json.load(fh).get("evals", [])

    def _save(self) -> None:
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fpath, "w", encoding="utf-8") as fh:
            json.dump({"evals": self._evals}, fh, ensure_ascii=False, indent=2)

    def record(
        self,
        eval_id: str,
        candidate_id: str,
        benchmark: str,
        split: str,
        dataset_version: str,
        manifest_digest: str,
        reward_mean: Optional[float],
        total_tokens: int,
        experiment_profile_id: Optional[str] = None,
        *,
        candidate_digest: Optional[str] = None,
        environment_digest: Optional[str] = None,
        evaluation_digest: Optional[str] = None,
        manifest_schema_version: int = 1,
    ) -> None:
        self._evals.append(
            {
                "eval_id": eval_id,
                "candidate_id": candidate_id,
                "benchmark": benchmark,
                "split": split,
                "dataset_version": dataset_version,
                "manifest_schema_version": manifest_schema_version,
                "candidate_digest": candidate_digest,
                "manifest_digest": manifest_digest,
                "environment_digest": environment_digest,
                "evaluation_digest": evaluation_digest or manifest_digest,
                "experiment_profile_id": experiment_profile_id,
                "reward_mean": reward_mean,
                "total_tokens": total_tokens,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save()

    def _matching(
        self,
        candidate_id: str,
        benchmark: str,
        split: str,
        dataset_version: str,
        evaluation_digest: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return [
            e
            for e in self._evals
            if e["candidate_id"] == candidate_id
            and e["benchmark"] == benchmark
            and e["split"] == split
            and e["dataset_version"] == dataset_version
            and (
                evaluation_digest is None or (e.get("evaluation_digest") or e["manifest_digest"]) == evaluation_digest
            )
        ]

    def overfit_stats(
        self,
        candidate_id: str,
        benchmark: str,
        split: str,
        dataset_version: str,
        evaluation_digest: Optional[str] = None,
    ) -> dict[str, Any]:
        """(candidate lineage, split, dataset_version, evaluation identity) eval count/trend (OF-3)."""
        records = self._matching(candidate_id, benchmark, split, dataset_version, evaluation_digest)
        return {
            "eval_count": len(records),
            "score_trend": score_trend_summary([r["reward_mean"] for r in records]),
        }

    def lineage_cost(self, candidate_id: str) -> dict[str, int]:
        records = [e for e in self._evals if e["candidate_id"] == candidate_id]
        return {"evals": len(records), "total_tokens": sum(int(e["total_tokens"]) for e in records)}

    def candidate_digests(self, candidate_id: str) -> set[str]:
        return {
            e["manifest_digest"] for e in self._evals if e["candidate_id"] == candidate_id and e["manifest_digest"]
        }

    def candidate_environment_digests(self, candidate_id: str) -> set[str]:
        return {
            e["environment_digest"]
            for e in self._evals
            if e["candidate_id"] == candidate_id and e.get("environment_digest")
        }

    def candidate_schema_versions(self, candidate_id: str) -> set[int]:
        return {int(e.get("manifest_schema_version", 1)) for e in self._evals if e["candidate_id"] == candidate_id}

    def bucketed_evals(self, candidate_id: str) -> dict[str, list[dict[str, Any]]]:
        """HF-3: group a lineage's evals by evaluation identity.

        Buckets keyed by schema + evaluation digest — cross-environment values are never merged;
        comparing across buckets is the caller's explicit, visible decision.
        """
        buckets: dict[str, list[dict[str, Any]]] = {}
        for e in self._evals:
            if e["candidate_id"] != candidate_id:
                continue
            key = "|".join(
                [
                    e["benchmark"],
                    e["split"],
                    e["dataset_version"],
                    str(e.get("experiment_profile_id")),
                    str(e.get("manifest_schema_version", 1)),
                    e.get("evaluation_digest") or e["manifest_digest"],
                    str(e.get("candidate_digest")),
                ]
            )
            buckets.setdefault(key, []).append(e)
        return buckets

    def bucket_summaries(self, candidate_id: str) -> list[dict[str, Any]]:
        summaries = []
        for key, records in self.bucketed_evals(candidate_id).items():
            rewards = [r["reward_mean"] for r in records if r["reward_mean"] is not None]
            summaries.append(
                {
                    "bucket": key,
                    "evals": len(records),
                    "reward_mean_latest": rewards[-1] if rewards else None,
                    "reward_mean_best": max(rewards) if rewards else None,
                    "score_trend": score_trend_summary([r["reward_mean"] for r in records]),
                    "total_tokens": sum(int(r["total_tokens"]) for r in records),
                    "manifest_schema_version": int(records[0].get("manifest_schema_version", 1)),
                    "harness_manifest_digest": records[0].get("manifest_digest"),
                    "candidate_digest": records[0].get("candidate_digest"),
                    "environment_digest": records[0].get("environment_digest"),
                    "evaluation_digest": records[0].get("evaluation_digest") or records[0].get("manifest_digest"),
                }
            )
        return summaries
