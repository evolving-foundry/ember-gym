# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""llm_sim — the caged content generator (design §3.2/§3.4, M2).

The **only** generative component of the office profile, and it is a *constrained
function*, never an agent (决议 2): one call per need, a frozen prompt from the
package's ``llm_sim.yaml``, temperature 0, guided decoding through the Responses
API ``text.format`` json_schema so the simulator can only emit content matching
the declared output schema. Grounding comes from the package's retrieval index
(keyword-overlap scoring, deterministic, top-k via ``fidelity.retrieval_top_k``).

The simulator model ref is config-declared and separated from the policy model
(决议 3, τ² 先例); the call rides the wheel-sanctioned server-side model path
(``call_judge`` over ``ServerClient.post``). Every call lands in the trace as
``(prompt_hash, seed, output)`` so drift is attributable (§3.2), and every call
is counted into the session's ``llm_sim_calls`` budget pipeline (决议 9).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Optional

from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

from resources_servers.synth_env.package_loader import canonical_json


class LlmSimError(RuntimeError):
    """The simulator call failed or returned unusable content — an environment defect."""


@dataclass
class SimTrace:
    """Attributable record of one simulator call (§3.2 漂移可归因)."""

    prompt_hash: str
    seed: int
    output: dict[str, Any]
    grounded_ids: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass(frozen=True)
class GroundingRow:
    id: str
    text: str
    tags: tuple[str, ...] = ()


def load_grounding(rows: list[dict[str, Any]]) -> list[GroundingRow]:
    grounding = []
    for row in rows:
        if not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
            raise LlmSimError(f"grounding row must carry string 'id' and 'text': {row!r}")
        grounding.append(
            GroundingRow(id=row["id"], text=row["text"], tags=tuple(row.get("tags") or []))
        )
    return grounding


def _terms(text: str) -> set[str]:
    # The query is often canonical JSON (no whitespace): split() would yield one
    # giant token, so tokenize on word characters / CJK runs instead.
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()))


def retrieve(grounding: list[GroundingRow], query: str, top_k: int) -> list[GroundingRow]:
    """Deterministic keyword-overlap retrieval (no embeddings, HF discipline)."""
    terms = _terms(query)

    def score(row: GroundingRow) -> tuple[int, str]:
        return (len(terms & _terms(row.text + " " + " ".join(row.tags))), row.id)

    ranked = sorted(grounding, key=score, reverse=True)
    return [row for row in ranked if score(row)[0] > 0][:max(top_k, 0)]


class LlmSimEngine:
    """Frozen-prompt, schema-guided simulator calls for one office session."""

    def __init__(
        self,
        *,
        llm_sim_config: dict[str, Any],
        grounding: list[GroundingRow],
        server_client: Any,
        model_server_name: str,
        seed: int,
        retrieval_top_k: int = 4,
        temperature: float = 0.0,
        max_output_tokens: int = 400,
        trace_sink: Optional[str] = None,
    ):
        self._config = llm_sim_config
        self._grounding = grounding
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._seed = seed
        self._top_k = retrieval_top_k
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        # Optional JSONL sink (design §3.2: every call lands in the trace so
        # drift is attributable). Resolved under the writable workspace by the
        # host — the sink never points into the read-only overlay.
        self._trace_sink = trace_sink
        self.traces: list[SimTrace] = []

    @property
    def output_schema(self) -> dict[str, Any]:
        return self._config.get("output_schema") or {"type": "object", "properties": {}, "required": []}

    def _render_prompt(self, state_summary: str, action: dict[str, Any], context_text: str) -> str:
        template = self._config.get("prompt_template")
        if not isinstance(template, str) or not template:
            raise LlmSimError("llm_sim.yaml must declare a non-empty 'prompt_template'")
        return template.format(state=state_summary, action=canonical_json(action), context=context_text)

    async def generate(
        self, state_summary: str, action: dict[str, Any], context: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], SimTrace]:
        """One constrained simulator call; returns ``(content, trace)``."""
        context = context or {}
        grounded = retrieve(self._grounding, f"{action.get('name', '')} {canonical_json(context)}", self._top_k)
        grounded_text = "\n\n".join(f"[{row.id}] {row.text}" for row in grounded) or "(no grounding hits)"
        prompt = self._render_prompt(state_summary, action, grounded_text)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        # The wheel's Responses create-params model has no `seed` field (and the
        # wheel is immutable, 决议 10), so the session seed rides `metadata` for
        # wire-level attribution. Determinism itself comes from the frozen
        # prompt, temperature 0, and the deterministic retrieval — never from
        # trusting the backend to honor a seed.
        create_params = NeMoGymResponseCreateParamsNonStreaming(
            model=self._model_server_name,
            input=[
                {"role": "system", "content": self._config.get("system", "You are a constrained content generator.")},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            metadata={"synth_env_seed": str(self._seed)},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "sim_output",
                    "strict": True,
                    "schema": self.output_schema,
                }
            },
        )
        started = monotonic()
        response = await call_judge(
            self._server_client,
            server_name=self._model_server_name,
            url_path="/v1/responses",
            json=create_params,
            response_model=NeMoGymResponse,
        )
        output = self._extract_output(response)
        self._validate_output(output)
        trace = SimTrace(
            prompt_hash=prompt_hash,
            seed=self._seed,
            output=output,
            grounded_ids=[row.id for row in grounded],
            duration_ms=round((monotonic() - started) * 1000, 3),
        )
        self.traces.append(trace)
        self._write_trace(trace)
        return output, trace

    def _write_trace(self, trace: SimTrace) -> None:
        """Append one JSONL trace line to the workspace sink (best effort never
        silent: a sink failure is an environment defect and propagates)."""
        if self._trace_sink is None:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": trace.prompt_hash,
            "seed": trace.seed,
            "duration_ms": trace.duration_ms,
            "grounded_ids": trace.grounded_ids,
            "output": trace.output,
        }
        with Path(self._trace_sink).open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _extract_output(self, response: NeMoGymResponse) -> dict[str, Any]:
        for item in reversed(response.output or []):
            if getattr(item, "type", None) != "message":
                continue
            for part in reversed(getattr(item, "content", None) or []):
                if getattr(part, "type", None) == "output_text":
                    try:
                        parsed = json.loads(part.text)
                    except json.JSONDecodeError as error:
                        raise LlmSimError(f"simulator output is not valid JSON: {error}") from error
                    if not isinstance(parsed, dict):
                        raise LlmSimError(f"simulator output must be a JSON object, got {type(parsed).__name__}")
                    return parsed
        raise LlmSimError("simulator response carried no output_text")

    def _validate_output(self, output: dict[str, Any]) -> None:
        schema = self.output_schema
        required = schema.get("required") or []
        missing = [name for name in required if name not in output]
        if missing:
            raise LlmSimError(f"simulator output is missing required fields {missing} (guided decoding)")
