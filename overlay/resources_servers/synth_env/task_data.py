# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-row schemas for the synth_env synthetic environment host (design C-SE-1 / C-EA-1).

A task row is one line of a synth environment package's ``tasks.jsonl`` (and of the
benchmark ``data/*.jsonl`` generated from it). The package is the single source of
truth; benchmark data files are a published thin shell of these rows (design §3.6).
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EnvReleasePin(BaseModel):
    """The triple pin that binds a session to one immutable asset package version.

    ``version`` accepts ints in JSON (design §3.6 shows ``"version": 3``) and is
    normalized to a string so pin comparisons are always textual.
    """

    model_config = ConfigDict(extra="forbid")

    ref_id: str
    version: str
    fingerprint: str

    @field_validator("version", "fingerprint", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return value

    def matches(self, other: "EnvReleasePin") -> bool:
        return (
            self.ref_id == other.ref_id
            and self.version == other.version
            and self.fingerprint == other.fingerprint
        )


class VerifierSpec(BaseModel):
    """A frozen verify predicate from the asset package (design §3.8).

    M1 implements ``artifact_compare``; the other kinds are declared so that
    packages carrying them fail loudly at load instead of silently mis-scoring.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SuccessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifier_spec: VerifierSpec


class TaskRow(BaseModel):
    """One synthetic environment task (design §3.6)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    env_release: EnvReleasePin
    goal: str
    success_check: SuccessCheck
    max_steps: int = Field(ge=1)
    seed: int = 0
    # Reserved for M2 (office profile); M1 tasks leave it empty.
    initial_actions: Optional[list[dict[str, Any]]] = None
