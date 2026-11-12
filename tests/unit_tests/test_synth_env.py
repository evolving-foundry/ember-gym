# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the synth_env synthetic environment host + terminal-loop agent.

Server side runs the full FastAPI chain (seed -> step -> verify, admin plane,
MCP harvesting) through ``TestClient`` against in-memory asset packages. Agent
side stubs both seams the real rollout traffic uses — ``ServerClient.post`` for
the model and the environment — exactly like the arc_agi_v3 tests. No real
model, no real ports for the agent. Terminal/office packages never execute
commands; the one exception is the M3 hybrid group (TestHybridDynamics), whose
packages allowlist a few harmless coreutils (cat/cp/ls/seq/sleep) that really
run inside per-step temp directories.
"""

import asyncio
import base64
import io
import json
import zipfile
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.mcp_auto_exposure import harvest_tools
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.server_utils import ServerClient
from resources_servers.synth_env.app import SynthEnvServer, SynthEnvServerConfig
from resources_servers.synth_env.package_loader import (
    PackageBusyError,
    PackageStore,
    SynthEnvPackageError,
    compute_fingerprint,
    load_package_zip,
)
from resources_servers.synth_env.task_data import EnvReleasePin
from responses_api_agents.synth_env_agent.app import (
    SynthEnvAgent,
    SynthEnvAgentConfig,
    SynthEnvAgentRunRequest,
    SynthEnvMalformedActionError,
    SynthEnvVerifyResponse,
    _parse_action,
)


# ---- deterministic demo package (self-contained twin of build_package.py) ------

REF_ID = "synth-terminal-demo"
README_V1 = "version: 1.0.0\n"
README_V2 = "version: 2.0.0\n"
VERSION_TXT = "1.2.0\n"

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
    "danger_patterns": [r"rm\s+-rf\s+/"],
    "state_schema": {},
}


def _demo_files(readme: str) -> dict[str, str]:
    replay = [
        {"cwd": "/", "cmd": "cat /home/user/README.md", "stdout": readme.strip()},
        {
            "cwd": "/",
            "cmd": "echo 1.2.0 > /home/user/version.txt",
            "stdout": "",
            "fs_writes": {"/home/user/version.txt": VERSION_TXT},
            "done": True,
        },
    ]
    fs_rows = [{"path": "/home/user/README.md", "content": readme}]
    return {
        "env_spec.json": json.dumps(ENV_SPEC),
        "replay/replay.jsonl": "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in replay),
        "fs/snapshot.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fs_rows),
    }


def _task_rows(ref_id: str, version: str, fingerprint: str) -> list[dict]:
    def task(task_id: str, goal: str, expect_files: dict[str, str], max_steps: int) -> dict:
        return {
            "task_id": task_id,
            "env_release": {"ref_id": ref_id, "version": version, "fingerprint": fingerprint},
            "goal": goal,
            "success_check": {
                "verifier_spec": {"kind": "artifact_compare", "payload": {"expect_files": expect_files}}
            },
            "max_steps": max_steps,
            "seed": 42,
        }

    return [
        task("t-win", "Write the upgraded version to /home/user/version.txt.", {"/home/user/version.txt": VERSION_TXT}, 5),
        task("t-cap", "This task cannot be completed within its step budget.", {"/home/user/never.txt": "x"}, 2),
    ]


def _zip_bytes(
    ref_id: str, version: str, fingerprint: str, files: dict[str, str], tasks: list[dict], profile: str = "terminal",
    manifest_extra: dict | None = None,
) -> bytes:
    manifest = {
        "ref_id": ref_id,
        "version": version,
        "fingerprint": fingerprint,
        "profile": profile,
        "description": "unit-test package",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("tasks.jsonl", "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks))
    return buffer.getvalue()


def _demo_package(version: str = "1", readme: str = README_V1) -> bytes:
    files = _demo_files(readme)
    fingerprint = compute_fingerprint(REF_ID, version, "terminal", files)
    return _zip_bytes(REF_ID, version, fingerprint, files, _task_rows(REF_ID, version, fingerprint))


# ---- server helpers ------------------------------------------------------------


def _server(*packages: bytes, **config_overrides) -> SynthEnvServer:
    server_client = MagicMock(spec=ServerClient)
    # Empty mapping => observability/token-capture correlation stays disabled.
    server_client.global_config_dict = {}
    config = SynthEnvServerConfig(
        host="0.0.0.0", port=8080, entrypoint="app.py", name="synth_env", package_paths=[], **config_overrides
    )
    server = SynthEnvServer(config=config, server_client=server_client)
    for package in packages:
        server._store.load(load_package_zip(package))
    return server


def _client(server: SynthEnvServer) -> TestClient:
    return TestClient(server.setup_webserver())


def _pin(package_bytes: bytes) -> EnvReleasePin:
    return load_package_zip(package_bytes).pin


def _bash(command: str) -> dict:
    return {"name": "bash", "args": {"command": command}}


def _seed_body(pin: EnvReleasePin, task_id: str = "t-win", max_steps: int | None = None, budgets: dict | None = None,
               fidelity: dict | None = None) -> dict:
    body: dict = {
        "task_id": task_id,
        "env_release": pin.model_dump(),
        "seed": 0,
        "fidelity": fidelity or {},
        "budgets": budgets or {},
    }
    if max_steps is not None:
        body["max_steps"] = max_steps
    return body


def base64_of(zip_bytes: bytes) -> str:
    return base64.b64encode(zip_bytes).decode()


def _nemo_response_dict() -> dict:
    return {
        "id": "resp_test",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": [],
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }


def _verify_body() -> dict:
    return {"responses_create_params": {"model": "policy", "input": []}, "response": _nemo_response_dict()}


def _chat_completion(command: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
                    ],
                },
            }
        ],
        "created": 0,
        "model": "policy",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


# ---- 组1: action parsing -------------------------------------------------------


class TestParseAction:
    def test_valid_bash_tool_call(self):
        action = _parse_action(_chat_completion("cat /home/user/README.md"))
        assert action == {"name": "bash", "args": {"command": "cat /home/user/README.md"}}

    def test_unknown_tool_name_is_malformed(self):
        response = _chat_completion("ls")
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "python"
        with pytest.raises(SynthEnvMalformedActionError, match="expected bash"):
            _parse_action(response)

    def test_invalid_json_arguments_is_malformed(self):
        response = _chat_completion("ls")
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
        with pytest.raises(SynthEnvMalformedActionError, match="invalid JSON"):
            _parse_action(response)

    def test_empty_command_is_malformed(self):
        for bad in ({"command": ""}, {"command": "   "}, {}):
            response = _chat_completion("ls")
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(bad)
            with pytest.raises(Exception, match="command"):
                _parse_action(response)

    def test_text_json_fallback_and_plain_text_malformed(self):
        response = {"choices": [{"message": {"role": "assistant", "content": 'Run this: {"command": "pwd"}'}}]}
        assert _parse_action(response) == {"name": "bash", "args": {"command": "pwd"}}
        plain = {"choices": [{"message": {"role": "assistant", "content": "I will look around now."}}]}
        with pytest.raises(Exception, match="no JSON object"):
            _parse_action(plain)


# ---- 组2: package loading ------------------------------------------------------


class TestPackageLoading:
    def test_tampered_content_rejected_by_fingerprint(self):
        files = _demo_files(README_V1)
        fingerprint = compute_fingerprint(REF_ID, "1", "terminal", files)
        tampered = dict(files)
        tampered["replay/replay.jsonl"] = files["replay/replay.jsonl"].replace("1.0.0", "9.9.9")
        with pytest.raises(SynthEnvPackageError, match="fingerprint mismatch"):
            load_package_zip(_zip_bytes(REF_ID, "1", fingerprint, tampered, _task_rows(REF_ID, "1", fingerprint)))

    def test_triple_pin_gate(self):
        package_bytes = _demo_package()
        loaded = load_package_zip(package_bytes)
        wrong = EnvReleasePin(ref_id="other-ref", version="1", fingerprint=loaded.manifest.fingerprint)
        with pytest.raises(SynthEnvPackageError, match="pin mismatch"):
            load_package_zip(package_bytes, expect=wrong)
        pinned = load_package_zip(package_bytes, expect=loaded.pin)
        # A matching pin registers the action schema the profile will enforce.
        assert pinned.env_spec.action_schema["name"] == "bash"

    def test_task_pin_mismatch_rejected(self):
        files = _demo_files(README_V1)
        fingerprint = compute_fingerprint(REF_ID, "1", "terminal", files)
        tasks = _task_rows(REF_ID, "1", "f" * 16)
        with pytest.raises(SynthEnvPackageError, match="pins fingerprint"):
            load_package_zip(_zip_bytes(REF_ID, "1", fingerprint, files, tasks))

    def test_package_immutability_and_store_discipline(self):
        package = load_package_zip(_demo_package())
        with pytest.raises(FrozenInstanceError):
            package.manifest = package.manifest  # noqa: B018 -- reassigning a frozen field
        with pytest.raises(ValidationError):
            package.manifest.version = "999"

        store = PackageStore()
        store.load(package)
        with pytest.raises(SynthEnvPackageError, match="already loaded"):
            store.load(package)
        ref, version = package.manifest.ref_id, package.manifest.version
        with pytest.raises(PackageBusyError):
            store.unload(ref, version, [(ref, version), ("other", "1"), (ref, version)])
        store.unload(ref, version, [])
        assert store.list() == []


# ---- 组3: terminal profile + replay over the full FastAPI chain -----------------


class TestTerminalFullChain:
    def test_seed_contract_rejections_are_explicit_4xx(self):
        server = _server(_demo_package())
        with _client(server) as client:
            # Unseeded traffic is refused, never silently turned into a session.
            assert client.post("/step", json={"action": _bash("ls")}).status_code == 404
            assert client.post("/verify", json=_verify_body()).status_code == 404
            # Unknown fidelity knobs are physically rejected (extra=forbid whitelist).
            pin = _pin(_demo_package())
            bad_knobs = _seed_body(pin)
            bad_knobs["fidelity"] = {"bogus_knob": 1}
            assert client.post("/seed_session", json=bad_knobs).status_code == 422
            # Unknown task id in a pinned package.
            assert client.post("/seed_session", json=_seed_body(pin, task_id="ghost")).status_code == 404

    def test_win_path_scores_one(self):
        server = _server(_demo_package())
        with _client(server) as client:
            seed = client.post("/seed_session", json=_seed_body(_pin(_demo_package()))).json()
            assert seed["initial"]["actions_available"] == ["bash"]
            assert seed["initial"]["obs"]["text"].startswith("TASK:")

            step = client.post("/step", json={"action": _bash("echo 1.2.0 > /home/user/version.txt")}).json()
            assert step["done"] is True

            verify = client.post("/verify", json=_verify_body())
            assert verify.status_code == 200
            body = verify.json()
            assert body["reward"] == 1.0
            assert body["reward_components"] == {"task_complete": 1.0}
            assert body["task_id"] == "t-win"
            assert body["steps"] == 1
            assert body["budget_exceeded"] is False

    def test_replay_miss_is_observable_and_session_continues(self):
        server = _server(_demo_package())
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(_demo_package())))
            miss = client.post("/step", json={"action": _bash("curl http://example.com")}).json()
            assert miss["done"] is False
            assert "[exit_code=127]" in miss["obs"]["text"]
            assert "no replay entry" in miss["obs"]["text"]

            # The session survives the miss; a recorded command still hits the trie.
            follow_up = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert follow_up["done"] is False
            assert follow_up["obs"]["text"] == "version: 1.0.0"
            assert client.post("/verify", json=_verify_body()).json()["reward"] == 0.0

    def test_step_cap_truncates_with_double_insurance(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_seed_body(_pin(package_bytes), task_id="t-cap", max_steps=100)).json()
            # The environment-side cap is min(request, package task row) = 2.
            detail = client.get(f"/admin/sessions/{seed['session_id']}").json()
            assert detail["max_steps"] == 2

            first = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert first["done"] is False
            second = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert second["done"] is True

            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 0.0
            assert verify["truncated"] is True
            assert verify["steps"] == 2

    def test_dual_sessions_are_isolated(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        pin = _pin(package_bytes)
        with _client(server) as writer, _client(server) as bystander:
            writer.post("/seed_session", json=_seed_body(pin))
            bystander.post("/seed_session", json=_seed_body(pin))
            writer.post("/step", json={"action": _bash("echo 1.2.0 > /home/user/version.txt")})
            assert writer.post("/verify", json=_verify_body()).json()["reward"] == 1.0
            assert bystander.post("/verify", json=_verify_body()).json()["reward"] == 0.0

    def test_dangerous_command_blocked_but_session_usable(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            # /bash is the profile-semantic route (MCP tool name "bash").
            blocked = client.post("/bash", json={"command": "rm -rf /"}).json()
            assert "blocked by policy" in blocked["obs"]["text"]
            assert blocked["info"]["violations"] == ["dangerous_command_blocked"]
            assert blocked["done"] is False

            step = client.post("/step", json={"action": _bash("echo 1.2.0 > /home/user/version.txt")}).json()
            assert step["done"] is True
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 1.0  # a blocked attempt does not zero a completed task
            assert verify["violations"] == 1


# ---- 组6: MCP auto-exposure -----------------------------------------------------


class TestMcpExposure:
    def test_admin_routes_filtered_action_tools_exposed(self):
        server = _server(_demo_package(), expose_tools_over_mcp=True)
        tools = harvest_tools(server.setup_webserver(), server)
        assert set(tools) == {"bash", "step", "reset"}
        assert all(tool.binding is not None for tool in tools.values())


# ---- 组7: admin plane -----------------------------------------------------------


class TestAdminPlane:
    def test_runtime_load_validates_pins_and_journals(self):
        server = _server(_demo_package())
        files = _demo_files(README_V2)
        fingerprint = compute_fingerprint(REF_ID, "2", "terminal", files)
        good_v2 = _zip_bytes(REF_ID, "2", fingerprint, files, _task_rows(REF_ID, "2", fingerprint))

        with _client(server) as client:
            # Tampered bytes: rejected with the integrity reason, nothing loaded.
            tampered = dict(files)
            tampered["env_spec.json"] = files["env_spec.json"].replace("2000", "100")
            bad = client.post(
                "/admin/packages/load",
                json={"zip_base64": base64_of(_zip_bytes(REF_ID, "2", fingerprint, tampered, _task_rows(REF_ID, "2", fingerprint)))},
            )
            assert bad.status_code == 422
            assert "fingerprint mismatch" in bad.json()["detail"]

            # A pin expectation that the package does not satisfy: rejected.
            wrong_expect = {"ref_id": REF_ID, "version": "2", "fingerprint": "0" * 16}
            mismatch = client.post("/admin/packages/load", json={"zip_base64": base64_of(good_v2), "expect": wrong_expect})
            assert mismatch.status_code == 422
            assert "pin mismatch" in mismatch.json()["detail"]

            loaded = client.post("/admin/packages/load", json={"zip_base64": base64_of(good_v2)})
            assert loaded.status_code == 200
            listed = client.get("/admin/packages").json()
            assert [p["version"] for p in listed if p["ref_id"] == REF_ID] == ["1", "2"]
            assert "load" in [entry["action"] for entry in server._journal.entries]

    def test_unload_refuses_active_session_then_allows_after_terminate(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_seed_body(_pin(package_bytes))).json()
            busy = client.post("/admin/packages/unload", json={"ref_id": REF_ID, "version": "1"})
            assert busy.status_code == 409

            terminated = client.post(
                f"/admin/sessions/{seed['session_id']}/terminate", json={"reason": "operator reset"}
            )
            assert terminated.status_code == 200
            unloaded = client.post("/admin/packages/unload", json={"ref_id": REF_ID, "version": "1"})
            assert unloaded.status_code == 200
            actions = [entry["action"] for entry in server._journal.entries]
            assert "terminate" in actions and "unload" in actions

    def test_dual_version_coexistence_and_session_binding(self):
        v1, v2 = _demo_package("1", README_V1), _demo_package("2", README_V2)
        server = _server(v1, v2)
        with _client(server) as client_a, _client(server) as client_b:
            client_a.post("/seed_session", json=_seed_body(_pin(v1)))
            client_b.post("/seed_session", json=_seed_body(_pin(v2)))
            out_a = client_a.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            out_b = client_b.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            # Each session replays the package version it pinned at seed time.
            assert out_a["obs"]["text"] == "version: 1.0.0"
            assert out_b["obs"]["text"] == "version: 2.0.0"

    def test_terminate_and_reap_journal_and_clear_sessions(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_seed_body(_pin(package_bytes))).json()
            reaped = client.post("/admin/sessions/reap", json={"ttl_s": 1e-6}).json()
            assert seed["session_id"] in reaped["reaped"]
            assert client.get(f"/admin/sessions/{seed['session_id']}").status_code == 404
            actions = [entry["action"] for entry in server._journal.entries]
            assert "reap" in actions

    def test_wall_clock_budget_forces_done_and_scores_zero(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes), budgets={"wall_clock_s": 1e-9}))
            step = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert step["done"] is True
            assert step["info"]["failure_class"] == "synth_env_budget_exceeded"
            # Idempotent: further steps report the forced-done state, no execution.
            again = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert again["done"] is True

            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 0.0
            assert verify["budget_exceeded"] is True
            assert "synth_env_budget_exceeded" in verify["failure_reason"]
            assert "budget_exceeded" in [entry["action"] for entry in server._journal.entries]

    def test_drain_rejects_new_sessions(self):
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            assert client.post("/admin/drain").json()["draining"] is True
            assert client.post("/seed_session", json=_seed_body(_pin(package_bytes))).status_code == 503
            assert client.get("/admin/metrics").json()["draining"] is True


# ---- 组4b: admin metrics 补齐 + /aggregate_metrics 契约槽 (§3.13) -------------------


class TestAdminMetricsAndAggregateSlot:
    def test_admin_metrics_latency_percentiles_and_drift(self):
        # Grounded generation: every call lands in the latency percentiles and
        # the drift proxy stays zero.
        grounded_package = _office_package(npc_mode="llm_sim", include_llm_sim=True, grounding=True)
        server = _server(grounded_package, llm_sim_model_server={"type": "responses_api_models", "name": "sim_model"})
        async def post(**kwargs):
            return _FakeResponse(_sim_response({"content": "已按两票制度复核通过"}))
        server.server_client.post = AsyncMock(side_effect=post)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(grounded_package)))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}})
            metrics = client.get("/admin/metrics").json()
        assert metrics["llm_sim_latency_ms"]["count"] == 1
        assert metrics["llm_sim_latency_ms"]["p50"] >= 0.0
        assert metrics["llm_sim_latency_ms"]["p95"] >= metrics["llm_sim_latency_ms"]["p50"]
        assert metrics["llm_sim_ungrounded_calls"] == 0
        assert metrics["drift_rate"] == 0.0

        # Ungrounded generation (no grounding index in the package): the call
        # counts as a drift-risk signal (§7.1 漂移监测 v1).
        ungrounded_package = _office_package(npc_mode="llm_sim", include_llm_sim=True)
        server = _server(ungrounded_package, llm_sim_model_server={"type": "responses_api_models", "name": "sim_model"})
        server.server_client.post = AsyncMock(side_effect=post)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(ungrounded_package)))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}})
            metrics = client.get("/admin/metrics").json()
        assert metrics["llm_sim_ungrounded_calls"] == 1
        assert metrics["drift_rate"] == 1.0

    def test_package_list_loaded_at_and_detail_inventory_counts(self):
        package_bytes = _office_package(grounding=True)
        server = _server(package_bytes)
        with _client(server) as client:
            listing = client.get("/admin/packages").json()
            assert listing[0]["loaded_at"] is not None
            detail = client.get(f"/admin/packages/{OFFICE_REF_ID}").json()
            version_detail = detail["1"]
            # One fsm + one assertion; one grounding row (§3.13 包详情字段).
            assert version_detail["rules_count"] == 2
            assert version_detail["grounding_rows"] == 1

    def test_aggregate_metrics_contract_slot_carries_synth_env_aggregates(self):
        # Masked samples never dilute the aggregates (scored-subset discipline).
        verify_rows = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "violations": 0, "steps": 2,
             "truncated": False, "budget_exceeded": False, "llm_sim_calls": 3},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0, "violations": 2, "steps": 4,
             "truncated": True, "budget_exceeded": True, "llm_sim_calls": 1},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "reward": 0.0, "violations": 100, "steps": 100,
             "truncated": True, "budget_exceeded": True, "llm_sim_calls": 99, "mask_sample": True},
        ]
        package_bytes = _demo_package()
        server = _server(package_bytes)
        with _client(server) as client:
            response = client.post("/aggregate_metrics", json={"verify_responses": verify_rows}).json()
        text = json.dumps(response)
        assert "synth_env_violations_mean" in text
        assert "synth_env_steps_mean" in text
        assert "synth_env_truncated_rate" in text
        assert "synth_env_budget_exceeded_rate" in text
        assert "synth_env_llm_sim_calls_mean" in text
        metrics = response["agent_metrics"]
        assert metrics["synth_env_violations_mean"] == 1.0
        assert metrics["synth_env_steps_mean"] == 3.0
        assert metrics["synth_env_truncated_rate"] == 0.5
        assert metrics["synth_env_budget_exceeded_rate"] == 0.5
        assert metrics["synth_env_llm_sim_calls_mean"] == 2.0
        # The key-metric slot surfaces exactly the synth_env aggregates (the
        # wheel additionally merges its own coverage/* stats into key_metrics).
        assert {k: v for k, v in response["key_metrics"].items() if k.startswith("synth_env_")} \
            == {k: v for k, v in metrics.items() if k.startswith("synth_env_")}


# ---- 组4: office 六件套 (design §3.2) --------------------------------------------
#
# Deterministic twin of build_package.py's office package: schema-checked ticket
# store, approval FSM + two-person assertion, sim-clock events, a whitelisted
# NPC supervisor, role visibility, and the llm_sim cage. NPC content runs in
# template mode unless a test opts into llm_sim with a stubbed model.

OFFICE_REF_ID = "office-power-test"
TICKET_ID = "T-1"
HIDDEN_NOTE = "supervisor-only note"

OFFICE_STATE_SCHEMA = {
    "tickets": {
        "key": "id",
        "fields": {
            "id": {"type": "string", "required": True},
            "status": {"type": "enum", "enum": ["open", "in_review", "approved", "rejected"], "required": True},
            "detail": {"type": "string"},
            "verified_by": {"type": "string"},
            "internal_note": {"type": "string", "visible_to": ["supervisor"]},
        },
    }
}

OFFICE_ACTIONS = [
    {
        "name": "review_ticket",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
        "effect": {"type": "fsm", "collection": "tickets", "key_from": "ticket_id"},
    },
    {
        "name": "verify_ticket",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "verifier": {"type": "string"}},
            "required": ["ticket_id", "verifier"],
        },
        "effect": {
            "type": "patch",
            "collection": "tickets",
            "key_from": "ticket_id",
            "fields": {"verified_by": {"from_arg": "verifier"}},
        },
    },
    {
        "name": "approve",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
        "effect": {"type": "fsm", "collection": "tickets", "key_from": "ticket_id"},
    },
    {
        "name": "write_report",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        "effect": {"type": "fs_write", "path_from": "path", "content_from": "content"},
    },
]

OFFICE_RULES = {
    "fsms": {
        "ticket_lifecycle": {
            "collection": "tickets",
            "field": "status",
            "transitions": [
                {"from": "open", "to": "in_review", "action": "review_ticket", "roles": ["agent"]},
                {"from": "in_review", "to": "approved", "action": "approve", "roles": ["supervisor"]},
            ],
        }
    },
    "assertions": [
        {
            "id": "two_person_rule",
            "collection": "tickets",
            "when": {"status": "approved"},
            "require": {"verified_by": {"not_empty": True}},
        }
    ],
}

OFFICE_LLM_SIM = {
    "system": "constrained generator",
    "prompt_template": "state:{state}\naction:{action}\ncontext:{context}",
    "output_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
    "max_output_tokens": 64,
}


def _office_files(
    *,
    npc_mode: str = "template",
    npc_allowed: list[str] | None = None,
    include_llm_sim: bool = False,
    grounding: bool = False,
    npc_auto_action: bool = False,
    extra_actions: list[dict] | None = None,
    perturbations: dict | None = None,
) -> dict[str, str]:
    auto_actions = [
        {
            "on": "event:chase_reminder",
            "when": {"collection": "tickets", "status": "in_review"},
            "action": {"name": "approve", "args": {"ticket_id": "{obj.id}"}},
        }
    ]
    if npc_auto_action:
        auto_actions.append(
            {
                "on": "action:review_ticket",
                "when": {"collection": "tickets", "status": "in_review"},
                "action": {"name": "approve", "args": {"ticket_id": "{obj.id}"}},
            }
        )
    npcs = {
        "npcs": [
            {
                "id": "supervisor",
                "role": "supervisor",
                "mode": npc_mode,
                "allowed_actions": npc_allowed if npc_allowed is not None else ["approve"],
                "template": "复核结果:{outcome}",
                "auto_actions": auto_actions,
            }
        ]
    }
    files = {
        "env_spec.json": json.dumps(
            {
                "action_schema": {"name": "office", "parameters": {"type": "object", "properties": {}}},
                "command_max_length": 2000,
                "danger_patterns": [],
                "state_schema": OFFICE_STATE_SCHEMA,
                "actions": OFFICE_ACTIONS + (extra_actions or []),
            }
        ),
        "fs/snapshot.jsonl": "",
        "state/seed.jsonl": json.dumps(
            {
                "kind": "tickets",
                "obj": {"id": TICKET_ID, "status": "open", "detail": "", "verified_by": "",
                        "internal_note": HIDDEN_NOTE},
            }
        ) + "\n",
        "rules.yaml": json.dumps(OFFICE_RULES),
        "events.yaml": json.dumps(
            {"step_minutes": 15,
             "events": [{"id": "chase_reminder", "at": 30,
                         "notify": {"npc": "supervisor", "message": "chase"}}]}
        ),
        "npcs.yaml": json.dumps(npcs),
    }
    if include_llm_sim:
        files["llm_sim.yaml"] = json.dumps(OFFICE_LLM_SIM)
    if perturbations is not None:
        files["perturbations.yaml"] = json.dumps(perturbations, ensure_ascii=False)
    if grounding:
        files["grounding/index.jsonl"] = (
            json.dumps({"id": "doc-1", "text": "tickets require a verifier before approval — two-person rule",
                        "tags": ["审批"]}) + "\n"
        )
    return files


def _office_task_rows(ref_id: str, version: str, fingerprint: str) -> list[dict]:
    return [
        {
            "task_id": "o-win",
            "env_release": {"ref_id": ref_id, "version": version, "fingerprint": fingerprint},
            "goal": "Approve ticket through the lifecycle.",
            "success_check": {
                "verifier_spec": {
                    "kind": "state_assert",
                    "payload": {
                        "checks": [
                            {
                                "collection": "tickets",
                                "match": {"id": TICKET_ID},
                                "expect": {"status": "approved", "verified_by": "王工"},
                            }
                        ]
                    },
                }
            },
            "max_steps": 6,
            "seed": 42,
        }
    ]


def _office_package(
    version: str = "1",
    *,
    manifest_profile: str = "office",
    sim_model: dict | None = None,
    **files_kwargs,
) -> bytes:
    files = _office_files(**files_kwargs)
    fingerprint = compute_fingerprint(OFFICE_REF_ID, version, manifest_profile, files)
    manifest_extra = {"description": "unit-test office package"}
    if sim_model is not None:
        manifest_extra["sim_model"] = sim_model
    return _zip_bytes(OFFICE_REF_ID, version, fingerprint, files, _office_task_rows(OFFICE_REF_ID, version, fingerprint),
                      profile=manifest_profile, manifest_extra=manifest_extra)


def _office_pin(package_bytes: bytes) -> EnvReleasePin:
    return load_package_zip(package_bytes).pin


def _office_seed_body(pin: EnvReleasePin, task_id: str = "o-win", **overrides) -> dict:
    body = _seed_body(pin, task_id=task_id)
    body.update(overrides)
    return body


class TestOfficeSixPack:
    def test_state_schema_rejects_illegal_write_observably(self):
        # A declared action whose patch targets a field outside the state schema:
        # the schema engine rejects it observably — never a silent partial write.
        smuggled = [{
            "name": "force_flag",
            "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}},
                           "required": ["ticket_id"]},
            "effect": {"type": "patch", "collection": "tickets", "key_from": "ticket_id",
                       "fields": {"escalated": {"fixed": True}}},
        }]
        package_bytes = _office_package(extra_actions=smuggled)
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            bad = client.post("/step", json={"action": {"name": "force_flag", "args": {"ticket_id": TICKET_ID}}}).json()
            assert "rejected" in bad["obs"]["text"]
            assert "illegal_state_write" in bad["info"]["violations"]
            # The session survives; a legal write still lands.
            good = client.post(
                "/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}}
            ).json()
            assert good["obs"]["text"] == "updated tickets[T-1]"

    def test_fsm_overreach_blocked_and_folded_into_compliance(self):
        package_bytes = _office_package(npc_auto_action=True)
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            # Agent attempts a supervisor-only transition straight from `open`.
            overreach = client.post("/step", json={"action": {"name": "approve", "args": {"ticket_id": TICKET_ID}}}).json()
            assert "blocked by rule" in overreach["obs"]["text"]
            assert overreach["info"]["violations"] == ["no_fsm_transition:tickets:approve"]

            # Legal path still wins afterwards — violation recorded, not fatal (决议 5).
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            done = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()
            assert done["done"] is True
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 1.0
            assert verify["reward_components"]["task_complete"] == 1.0
            assert verify["reward_components"]["compliance"] == 0.75  # one recorded violation folded in

    def test_npc_overreach_refused_by_whitelist_and_rules(self):
        # The NPC card only whitelists "comment" yet its auto-action tries to
        # approve — the whitelist plus rules engine must refuse (越权必被拒).
        package_bytes = _office_package(npc_allowed=["comment"])
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}})
            # The chase event fires on the next step's clock tick (at 30 sim-minutes).
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            npc_events = client.get("/observe").json()["info"]["npc_events"]
            assert any("unauthorized_action:approve" in event for event in npc_events)
            # State untouched: the ticket is still in_review, not approved.
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 0.0

    def test_role_visibility_crops_hidden_fields(self):
        package_bytes = _office_package()
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes))).json()
            # The agent's observation never contains supervisor-only fields, but
            # the backing store keeps them (cropping is observation-layer duty).
            assert HIDDEN_NOTE not in seed["initial"]["obs"]["text"]
            assert HIDDEN_NOTE not in seed["initial"]["state_summary"]
            observed = client.get("/observe").json()
            assert HIDDEN_NOTE not in observed["state_summary"]

    def test_time_compression_preserves_event_order(self):
        package_bytes = _office_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes), fidelity={"time_compression": 4.0}))
            first = client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}}).json()
            # Clock advanced 60 sim-minutes in one step: the 30-minute reminder
            # already fired (order preserved, wall times squeezed — 保序不保时).
            assert "[event] chase_reminder" in first["obs"]["text"]


# ---- 组5: verify 多分量 (§3.8) -----------------------------------------------------


class TestOfficeVerify:
    def test_clean_win_scores_full_compliance_and_efficiency(self):
        package_bytes = _office_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            done = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()
            assert done["done"] is True
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward_components"] == {"task_complete": 1.0, "compliance": 1.0, "efficiency": round(1 - 2 / 6, 4)}

    def test_report_artifact_task_uses_artifact_compare(self):
        package_bytes = _office_package()
        files = _office_files()
        fingerprint = compute_fingerprint(OFFICE_REF_ID, "1", "office", files)
        rows = _office_task_rows(OFFICE_REF_ID, "1", fingerprint)
        rows[0]["success_check"] = {
            "verifier_spec": {"kind": "artifact_compare", "payload": {"expect_files": {"/r.md": "ok\n"}}}
        }
        package_bytes = _zip_bytes(OFFICE_REF_ID, "1", fingerprint, files, rows, profile="office")
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            done = client.post(
                "/step", json={"action": {"name": "write_report", "args": {"path": "/r.md", "content": "ok\n"}}}
            ).json()
            assert done["done"] is True
            assert client.post("/verify", json=_verify_body()).json()["reward"] == 1.0


# ---- 组5b: perturbation 旋钮 (§3.9, DataDesigner failure_probe/state_patch 词汇) ----


class TestFidelityPerturbation:
    def test_state_patch_applies_at_seed_through_schema(self):
        # state_patch 词汇 ({op, path, value}) mapped from DataDesigner: the patch
        # lands through the state schema before the first observation.
        perturbations = {"state_patch": [{"op": "replace", "path": ["tickets", TICKET_ID, "detail"], "value": "缺陷已现场确认"}]}
        package_bytes = _office_package(perturbations=perturbations)
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes), fidelity={"perturbation": "state_patch"})).json()
            assert "缺陷已现场确认" in seed["initial"]["obs"]["text"]

    def test_perturbation_off_leaves_seed_state_untouched(self):
        perturbations = {"state_patch": [{"op": "replace", "path": ["tickets", TICKET_ID, "detail"], "value": "缺陷已现场确认"}]}
        package_bytes = _office_package(perturbations=perturbations)
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes))).json()
            assert "缺陷已现场确认" not in seed["initial"]["obs"]["text"]

    def test_failure_probe_injected_observably_and_session_still_wins(self):
        # The probe consumes the turn, is observable in rule_events, is NOT a
        # violation (environment fault ≠ agent misbehavior), and the session
        # can still complete the task afterwards.
        perturbations = {"failure_probe": [{"at_step": 2, "action": "review_ticket", "message": "审批服务瞬断"}]}
        package_bytes = _office_package(perturbations=perturbations)
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes), fidelity={"perturbation": "failure_probe"}))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            probed = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()
            assert "failure_probe injected" in probed["obs"]["text"]
            assert "审批服务瞬断" in probed["obs"]["text"]
            assert "failure_probe:review_ticket@2" in probed["info"]["rule_events"]
            assert probed["info"]["violations"] == []
            # The retry at step 3 runs the real pipeline: review → in_review, the
            # chase event fires, and the (whitelisted) NPC supervisor approves.
            done = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()
            assert done["done"] is True
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 1.0
            assert verify["reward_components"]["compliance"] == 1.0

    def test_perturbation_knob_without_assets_rejected_loudly(self):
        # Neither the office package nor the terminal package declares
        # perturbations.yaml — a non-off knob must never degrade into a
        # silent no-op.
        office_bytes = _office_package()
        terminal_bytes = _demo_package()
        server = _server(office_bytes, terminal_bytes)
        with _client(server) as client:
            office_reject = client.post("/seed_session", json=_office_seed_body(_office_pin(office_bytes), fidelity={"perturbation": "state_patch"}))
            assert office_reject.status_code == 422
            assert "no perturbations.yaml" in office_reject.json()["detail"]
            terminal_reject = client.post("/seed_session", json=_seed_body(_pin(terminal_bytes), fidelity={"perturbation": "failure_probe"}))
            assert terminal_reject.status_code == 422

    def test_malformed_perturbations_rejected_at_load(self):
        # remove is owned by asset-side variant generation; runtime injection
        # rejects it at package load (loud, before any session can be seeded).
        files = _office_files(perturbations={"state_patch": [{"op": "remove", "path": ["tickets", TICKET_ID], "value": None}]})
        fingerprint = compute_fingerprint(OFFICE_REF_ID, "1", "office", files)
        package_bytes = _zip_bytes(OFFICE_REF_ID, "1", fingerprint, files,
                                   _office_task_rows(OFFICE_REF_ID, "1", fingerprint), profile="office")
        with pytest.raises(SynthEnvPackageError, match="remove is owned by asset-side"):
            load_package_zip(package_bytes)


# ---- 组5c: visibility / npc_autonomy 旋钮 (§3.9) ------------------------------------


class TestFidelityVisibilityAutonomy:
    def test_expose_all_observation_contains_role_hidden_fields(self):
        # visibility=expose_all crops nothing: the agent's observation surface
        # (initial obs + state_summary) includes supervisor-only fields.
        package_bytes = _office_package()
        server = _server(package_bytes)
        with _client(server) as client:
            seed = client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes), fidelity={"visibility": "expose_all"})).json()
            assert HIDDEN_NOTE in seed["initial"]["obs"]["text"]
            # Adjudication semantics never shift with the knob: a supervisor-only
            # transition attempted by the agent role is still blocked.
            blocked = client.post("/step", json={"action": {"name": "approve", "args": {"ticket_id": TICKET_ID}}}).json()
            assert "blocked by rule" in blocked["obs"]["text"]

    def test_npc_autonomy_loose_skips_whitelist_but_rules_still_adjudicate(self):
        # Empty card whitelist: under strict every NPC turn is refused outright;
        # under loose the nomination goes through and the *rules engine* decides.
        package_bytes = _office_package(npc_allowed=[], npc_auto_action=True)
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes), fidelity={"npc_autonomy": "loose"}))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            # Step 2: review → in_review; the chase event fires and the NPC's
            # approve is rule-permitted (supervisor role) despite the empty whitelist.
            done = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()
            assert "复核结果:tickets[T-1] approved" in done["obs"]["text"]
            npc_events = client.get("/observe").json()["info"]["npc_events"]
            assert any("approve:tickets:T-1->approved" in event for event in npc_events)
            assert not any("unauthorized_action" in event for event in npc_events)
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 1.0


# ---- 组9: hybrid 类真实动力学 (§3.4, M3) --------------------------------------------
# 这些包允许少数无害 coreutils 真实执行(每步物化临时目录,用完即毁);
# 其余命令走 replay 回落/显式拒绝——"安全子集真实执行 + 其余模拟"。

HYBRID_ALLOWLIST = ["cat", "cp", "ls", "seq", "sleep", "mkdir"]


def _hybrid_files(*, env_extra: dict | None = None, replay_entries: list[dict] | None = None) -> dict[str, str]:
    env_spec = dict(ENV_SPEC)
    env_spec.update({
        "dynamics": "hybrid",
        "real_exec_allowlist": HYBRID_ALLOWLIST,
        "exec_timeout_s": 3.0,
        "exec_max_output_chars": 200,
    })
    if env_extra:
        env_spec.update(env_extra)
    replay = replay_entries if replay_entries is not None else [
        {"cwd": "/", "cmd": "cat /home/user/README.md", "stdout": "REPLAY-STAND-IN"},
        {"cwd": "/", "cmd": "df -h", "stdout": "Filesystem  Size  Used  Use%\n/dev/sda1  40G  21G  53%\n"},
        {"cwd": "/", "cmd": "echo 1.2.0 > /home/user/version.txt", "stdout": "",
         "fs_writes": {"/home/user/version.txt": VERSION_TXT}, "done": True},
    ]
    fs_rows = [{"path": "/home/user/README.md", "content": README_V1}]
    return {
        "env_spec.json": json.dumps(env_spec),
        "replay/replay.jsonl": "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in replay),
        "fs/snapshot.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fs_rows),
    }


def _hybrid_package(version: str = "h1", expect_files: dict | None = None, **files_kwargs) -> bytes:
    files = _hybrid_files(**files_kwargs)
    fingerprint = compute_fingerprint(REF_ID, version, "terminal", files)
    rows = _task_rows(REF_ID, version, fingerprint)
    if expect_files is not None:
        rows = [dict(rows[0], success_check={"verifier_spec": {"kind": "artifact_compare",
                                                               "payload": {"expect_files": expect_files}}})]
    return _zip_bytes(REF_ID, version, fingerprint, files, rows)


class TestHybridDynamics:
    def test_allowlisted_command_really_executes_and_lands_in_fs(self):
        expect = {"/home/user/copy.md": README_V1}
        package_bytes = _hybrid_package(expect_files=expect)
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("cp /home/user/README.md /home/user/copy.md")}).json()
            # Real stdout (silent cp, exit 0) and the write lands: the diff channel
            # shows it, the artifact verifier scores it.
            assert "[exit_code" not in step["obs"]["text"]
            assert step["done"] is True
            observed = client.get("/observe").json()
            assert "copy.md" in observed["state_summary"]
            assert client.post("/verify", json=_verify_body()).json()["reward"] == 1.0

    def test_real_execution_takes_precedence_over_replay(self):
        package_bytes = _hybrid_package()  # replay table holds "REPLAY-STAND-IN" for cat
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            # The command ran for real (README content), not the recorded stand-in.
            assert step["obs"]["text"].startswith("version: 1.0.0")
            assert "REPLAY-STAND-IN" not in step["obs"]["text"]

    def test_allowlist_miss_falls_back_to_replay(self):
        package_bytes = _hybrid_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("df -h")}).json()
            # Not allowlisted, but recorded: replayed verbatim ("其余模拟", §3.4).
            assert "/dev/sda1" in step["obs"]["text"]

    def test_allowlist_miss_without_replay_entry_is_observable_127(self):
        package_bytes = _hybrid_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("curl http://example.invalid")}).json()
            assert "[exit_code=127]" in step["obs"]["text"]

    def test_shell_metachar_never_reaches_a_subprocess(self):
        package_bytes = _hybrid_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            # `echo` is allowlisted, but the redirection makes it shell syntax —
            # it must take the miss policy, never shell=True.
            step = client.post("/step", json={"action": _bash("echo pwned > /home/user/pwn.txt")}).json()
            assert "[exit_code=127]" in step["obs"]["text"]
            listing = client.post("/step", json={"action": _bash("ls /home/user")}).json()
            assert "pwn.txt" not in listing["obs"]["text"]

    def test_reject_policy_refuses_miss_outright(self):
        package_bytes = _hybrid_package(env_extra={"hybrid_miss_policy": "reject"})
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("df -h")}).json()
            assert "[exit_code=126]" in step["obs"]["text"]
            assert "outside the real-exec allowlist" in step["obs"]["text"]

    def test_timeout_kills_runaway_command(self):
        package_bytes = _hybrid_package(env_extra={"exec_timeout_s": 0.3})
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("sleep 5")}).json()
            assert "[exit_code=124]" in step["obs"]["text"]
            assert "timed out" in step["obs"]["text"]

    def test_output_truncation_is_capped_and_marked(self):
        package_bytes = _hybrid_package(env_extra={"exec_max_output_chars": 200})
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            step = client.post("/step", json={"action": _bash("seq 1 10000")}).json()
            assert len(step["obs"]["text"]) < 400
            assert "[output truncated]" in step["obs"]["text"]

    def test_dangerous_command_blocked_before_dynamics(self):
        package_bytes = _hybrid_package()
        server = _server(package_bytes)
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(package_bytes)))
            blocked = client.post("/step", json={"action": _bash("rm -rf /")}).json()
            assert "blocked by policy" in blocked["obs"]["text"]
            # The session survives and allowlisted real execution still works.
            good = client.post("/step", json={"action": _bash("ls /home/user")}).json()
            assert "README.md" in good["obs"]["text"]

    def test_replay_and_hybrid_versions_coexist_side_by_side(self):
        server = _server(_demo_package("1"), _hybrid_package("h1"))
        with _client(server) as client:
            client.post("/seed_session", json=_seed_body(_pin(_demo_package("1"))))
            replayed = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert replayed["obs"]["text"] == "version: 1.0.0"  # v1: replay semantics
            client.post("/seed_session", json=_seed_body(_pin(_hybrid_package("h1"))))
            real = client.post("/step", json={"action": _bash("cat /home/user/README.md")}).json()
            assert real["obs"]["text"].startswith("version: 1.0.0") and "REPLAY" not in real["obs"]["text"]


class _FakeResponse:
    """Just enough of aiohttp.ClientResponse for raise_for_status/get_response_json."""

    def __init__(self, payload: dict, cookies: dict | None = None):
        self._payload = json.dumps(payload).encode()
        self.ok = True
        self.status = 200
        self.cookies = cookies or {}

    def raise_for_status(self):
        return None

    async def read(self) -> bytes:
        return self._payload


def _static_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are an operator inside a synthetic terminal."},
        {"role": "user", "content": "Complete the task with the bash tool."},
    ]


# ---- 组8: llm_sim 内容笼子 (§3.2/§3.4, stubbed model) -------------------------------


def _sim_response(content: dict) -> dict:
    response = _nemo_response_dict()
    response["model"] = "sim_model"
    response["output"] = [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_sim",
            "content": [{"type": "output_text", "text": json.dumps(content, ensure_ascii=False), "annotations": []}],
        }
    ]
    return response


class TestLlmSimCage:
    def _sim_server(self, package_bytes: bytes, sim_calls: list[dict]):
        async def post(**kwargs):
            sim_calls.append(kwargs)
            return _FakeResponse(_sim_response({"content": "已按两票制度复核通过"}))

        server = _server(package_bytes, llm_sim_model_server={"type": "responses_api_models", "name": "sim_model"})
        server.server_client.post = AsyncMock(side_effect=post)
        return server

    def test_guided_decoding_payload_grounding_and_trace(self):
        package_bytes = _office_package(npc_mode="llm_sim", include_llm_sim=True, grounding=True)
        sim_calls: list[dict] = []
        server = self._sim_server(package_bytes, sim_calls)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            step = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()

            # The caged call: /v1/responses on the *simulator* ref, temperature 0,
            # json_schema-guided output, grounding context included.
            assert sim_calls, "llm_sim made no call"
            call = sim_calls[0]
            assert call["server_name"] == "sim_model"
            assert call["url_path"] == "/v1/responses"
            payload = call["json"].model_dump(mode="json")
            assert payload["temperature"] == 0.0
            # The wheel's Responses model has no seed field; the session seed
            # rides metadata for wire-level attribution (§3.2).
            assert payload["metadata"]["synth_env_seed"] == "0"
            fmt = payload["text"]["format"]
            assert fmt["type"] == "json_schema"
            assert fmt["schema"]["required"] == ["content"]
            assert "[doc-1] tickets require a verifier" in payload["input"][1]["content"]

            # NPC content came from the (stubbed) simulator, not the template.
            assert "已按两票制度复核通过" in step["obs"]["text"]
            assert step["info"]["usage"]["llm_sim_calls"] == 1

    def test_npc_sim_budget_exhaustion_forces_done(self):
        # Two NPC turns fire inside one step (action trigger + event trigger via a
        # 2x sim-clock, so the 30-minute chase lands in step 1); the session budget
        # caps simulator calls at 1 and the mid-step re-check forces done.
        package_bytes = _office_package(npc_mode="llm_sim", include_llm_sim=True, npc_auto_action=True)
        server = self._sim_server(package_bytes, [])
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes), budgets={"llm_sim_calls": 1},
                                                                fidelity={"time_compression": 2.0}))
            step = client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}}).json()
            assert step["done"] is True
            assert step["info"]["failure_class"] == "synth_env_budget_exceeded"
            assert "budget exhausted" in step["obs"]["text"]
            verify = client.post("/verify", json=_verify_body()).json()
            assert verify["reward"] == 0.0 and verify["budget_exceeded"] is True

    def test_trace_sink_writes_one_jsonl_line_per_call(self, tmp_path):
        # The workspace trace sink records every simulator call (prompt_hash,
        # seed, duration, grounding ids, output) — drift is attributable (§3.2).
        package_bytes = _office_package(npc_mode="llm_sim", include_llm_sim=True, grounding=True)
        sink = tmp_path / "sim_trace.jsonl"
        server = _server(package_bytes, llm_sim_model_server={"type": "responses_api_models", "name": "sim_model"},
                         llm_sim_trace_fpath=str(sink))
        async def post(**kwargs):
            return _FakeResponse(_sim_response({"content": "已按两票制度复核通过"}))
        server.server_client.post = AsyncMock(side_effect=post)
        with _client(server) as client:
            client.post("/seed_session", json=_office_seed_body(_office_pin(package_bytes)))
            client.post("/step", json={"action": {"name": "verify_ticket", "args": {"ticket_id": TICKET_ID, "verifier": "王工"}}})
            client.post("/step", json={"action": {"name": "review_ticket", "args": {"ticket_id": TICKET_ID}}})
        lines = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 1
        entry = lines[0]
        assert entry["seed"] == 0
        assert entry["prompt_hash"] and isinstance(entry["duration_ms"], float)
        assert entry["grounded_ids"] == ["doc-1"]
        assert entry["output"] == {"content": "已按两票制度复核通过"}
        # In-memory traces mirror the sink.
        assert any(s.llm is not None and len(s.llm.traces) == 1 for s in server._sessions.values())

    def test_package_pinned_sim_ref_mismatch_rejected(self):
        pinned = _office_package(npc_mode="llm_sim", include_llm_sim=True,
                                 sim_model={"type": "responses_api_models", "name": "pkg_sim"})
        server = _server(pinned, llm_sim_model_server={"type": "responses_api_models", "name": "sim_model"})
        with _client(server) as client:
            conflict = client.post("/seed_session", json=_office_seed_body(_office_pin(pinned)))
            assert conflict.status_code == 409
            assert "simulator model" in conflict.json()["detail"]

        unpinned = _office_package(npc_mode="llm_sim", include_llm_sim=True,
                                   sim_model={"type": "responses_api_models", "name": "pkg_sim"})
        bare = _server(unpinned)  # no host simulator configured at all
        with _client(bare) as client:
            unavailable = client.post("/seed_session", json=_office_seed_body(_office_pin(unpinned)))
            assert unavailable.status_code == 503


class TestAgentEndToEnd:
    def _agent(self, model_responses: list[dict], step_results: list[dict], verify_response: dict) -> SynthEnvAgent:
        async def post(**kwargs):
            if kwargs["server_name"] == "policy_model":
                return _FakeResponse(model_responses.pop(0))
            if kwargs["server_name"] == "synth_env":
                path = kwargs["url_path"]
                if path == "/seed_session":
                    return _FakeResponse(
                        {
                            "session_id": "sess-1",
                            "initial": {
                                "obs": {"channel": "text", "text": "TASK: write the version"},
                                "state_summary": "cwd=/",
                                "actions_available": ["bash"],
                            },
                        },
                        cookies={"nemo_gym_session_id": "sess-1"},
                    )
                if path == "/step":
                    return _FakeResponse(step_results.pop(0))
                if path == "/verify":
                    return _FakeResponse(verify_response)
            raise AssertionError(f"unexpected call: {kwargs['server_name']} {kwargs.get('url_path')}")

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client.post = AsyncMock(side_effect=post)
        config = SynthEnvAgentConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="app.py",
            name="synth_env_agent",
            resources_server={"type": "resources_servers", "name": "synth_env"},
            model_server={"type": "responses_api_models", "name": "policy_model"},
            history_window=4,
        )
        return SynthEnvAgent(config=config, server_client=server_client)

    def _run_body(self) -> SynthEnvAgentRunRequest:
        return SynthEnvAgentRunRequest(
            task_id="t-win",
            goal="Write the upgraded version to /home/user/version.txt.",
            env_release={"ref_id": REF_ID, "version": "1", "fingerprint": "0" * 16},
            max_steps=3,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                model="policy", input=_static_messages()
            ),
        )

    def test_full_loop_wins_and_verifies(self):
        async def scenario():
            agent = self._agent(
                model_responses=[_chat_completion("echo 1.2.0 > /home/user/version.txt")],
                step_results=[{"obs": {"channel": "text", "text": ""}, "reward_hint": None, "done": True, "info": {"failure_class": None}}],
                verify_response={
                    "reward": 1.0,
                    "reward_components": {"task_complete": 1.0},
                    "response": _nemo_response_dict(),
                    "responses_create_params": {"model": "policy", "input": []},
                    "task_id": "t-win",
                    "steps": 1,
                    "truncated": False,
                    "budget_exceeded": False,
                    "violations": 0,
                },
            )
            return await agent.run(request=None, body=self._run_body())

        response = asyncio.run(scenario())
        assert isinstance(response, SynthEnvVerifyResponse)
        assert response.reward == 1.0
        assert response.task_id == "t-win"

    def test_model_payload_carries_bash_tool_and_dataset_prompts(self):
        async def scenario():
            agent = self._agent(
                model_responses=[_chat_completion("cat /home/user/README.md")],
                step_results=[{"obs": {"channel": "text", "text": "ok"}, "reward_hint": None, "done": False, "info": {}}],
                verify_response=self._verify_ok(),
            )
            body = self._run_body()
            body.max_steps = 1  # one model turn is enough to inspect the payload
            await agent.run(request=None, body=body)
            model_calls = [c for c in agent.server_client.post.await_args_list if c.kwargs["server_name"] == "policy_model"]
            assert model_calls, "no model call recorded"
            return model_calls[0].kwargs["json"]

        payload = asyncio.run(scenario())
        assert payload["tools"][0]["function"]["name"] == "bash"
        system_texts = [m["content"] for m in payload["messages"] if m["role"] == "system"]
        assert system_texts == ["You are an operator inside a synthetic terminal."]

    def _verify_ok(self) -> dict:
        return {
            "reward": 0.0,
            "reward_components": {"task_complete": 0.0},
            "response": _nemo_response_dict(),
            "responses_create_params": {"model": "policy", "input": []},
            "task_id": "t-win",
            "steps": 1,
            "truncated": False,
            "budget_exceeded": False,
            "violations": 0,
        }

    def test_budget_exceeded_maps_to_failure_class(self):
        async def scenario():
            agent = self._agent(
                model_responses=[_chat_completion("cat /home/user/README.md")],
                step_results=[
                    {
                        "obs": {"channel": "text", "text": "budget exhausted"},
                        "reward_hint": None,
                        "done": True,
                        "info": {"failure_class": "synth_env_budget_exceeded"},
                    }
                ],
                verify_response=self._verify_ok(),
            )
            return await agent.run(request=None, body=self._run_body())

        response = asyncio.run(scenario())
        assert response.reward == 0.0
        assert getattr(response, NG_FAILURE_CLASS_KEY) == "synth_env_budget_exceeded"

    def test_malformed_action_maps_to_failure_class(self):
        async def scenario():
            agent = self._agent(
                model_responses=[{"choices": [{"message": {"role": "assistant", "content": "I will look around now."}}]}],
                step_results=[],
                verify_response=self._verify_ok(),
            )
            return await agent.run(request=None, body=self._run_body())

        response = asyncio.run(scenario())
        assert response.reward == 0.0
        assert getattr(response, NG_FAILURE_CLASS_KEY) == "synth_env_malformed_action"
        assert "no JSON object" in response.error
