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
"""Unit tests for the dual-protocol deployment server (eval_service.server).

HTTP routes and MCP tools are exercised through the real FastAPI app via TestClient; the Gym
subprocess runner is faked, so no server binaries are needed (same isolation as
test_eval_service.py).
"""

import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from agent_gym.layout import GymLayout
from eval_service import dataset
from eval_service import server as server_module
from eval_service.contract import (
    EVAL_TOOL_PARAMETERS,
    GYM_EVALUATE_TOOL_DESCRIPTION,
    GYM_EVALUATE_TOOL_NAME,
    EvalRequest,
    EvalStatus,
)
from eval_service.guards import Ledger
from eval_service.profile import AdapterSpec, BenchmarkSpec, ExperimentProfile
from eval_service.promotion import PromotionPipeline
from eval_service.runner import GymEvalRunner, RunOutcome
from eval_service.server import (
    FEEDBACK_TOOL_NAME,
    RESULT_TOOL_NAME,
    ServerConfig,
    _is_loopback_client,
    build_app,
)
from eval_service.service import EvaluationService


RPC_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def test_main_requires_overlay_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ServerConfig(
        profile_path=Path("profile.yaml"),
        results_root=Path("results"),
        ledger_path=Path("ledger.json"),
    )
    calls: dict[str, object] = {}
    layout = object()

    def fake_prepare_runtime(**kwargs: object) -> object:
        calls.update(kwargs)
        return layout

    monkeypatch.setattr(server_module, "_config_from_args", lambda: config)
    monkeypatch.setattr(server_module, "prepare_runtime", fake_prepare_runtime)
    monkeypatch.setattr(
        server_module,
        "build_app",
        lambda _config, layout=None: calls.setdefault("app_layout", layout),
    )
    monkeypatch.setattr(server_module.uvicorn, "run", lambda *_args, **_kwargs: None)

    server_module.main()

    assert calls["require_overlay"] is True
    assert calls["verify_trees"] is True
    assert calls["app_layout"] is layout


class FakeRunner(GymEvalRunner):
    """Writes Gym-shaped output JSONL; optionally gated for deterministic /pending checks."""

    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root=repo_root)
        self.calls = 0
        self.gate = threading.Event()
        self.gate.set()

    def execute(self, argv: list[str], timeout_s: float | None) -> RunOutcome:
        self.calls += 1
        output = Path(argv[argv.index("--output") + 1])
        self.gate.wait(timeout=10)
        dataset.write_jsonl([{"reward": 1.0, "ng_perf": {"total_tokens": 10}}, {"reward": 0.5}], output)
        return RunOutcome(EvalStatus.COMPLETED, [], 0, 0, 1.0, 0, "")


def make_profile(tmp_path: Path, with_holdout: bool = False) -> ExperimentProfile:
    dataset_fpath = tmp_path / "toy_train.jsonl"
    dataset.write_jsonl(
        [
            {
                "task_id": f"toy-{i}",
                "responses_create_params": {"input": [{"role": "user", "content": f"q{i}"}]},
                "verifier_metadata": {},
            }
            for i in range(10)
        ],
        dataset_fpath,
    )
    holdout_fpath = None
    if with_holdout:
        holdout_fpath = tmp_path / "toy_holdout.jsonl"
        dataset.write_jsonl(
            [
                {
                    "task_id": f"hold-{i}",
                    "responses_create_params": {"input": [{"role": "user", "content": f"h{i}"}]},
                    "verifier_metadata": {},
                }
                for i in range(3)
            ],
            holdout_fpath,
        )
    return ExperimentProfile(
        profile_id="profile-test",
        adapters={"claude_code": AdapterSpec(harness_agent="claude_code_agent", version="1")},
        benchmarks={
            "toy": BenchmarkSpec(
                dataset_fpath=str(dataset_fpath),
                holdout_dataset_fpath=str(holdout_fpath) if holdout_fpath else None,
                dataset_version="v1",
                probe_split_size=4,
            )
        },
    )


def make_config(tmp_path: Path, profile: ExperimentProfile) -> ServerConfig:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile.model_dump(mode="json")), encoding="utf-8")
    return ServerConfig(
        profile_path=profile_path,
        results_root=tmp_path / "results",
        ledger_path=tmp_path / "ledger.json",
        journal_path=tmp_path / "journal.json",
        reports_root=tmp_path / "reports",
        repo_root=tmp_path,
        gym_commit="test-commit",
    )


def make_request(**overrides) -> EvalRequest:
    payload = {
        "candidate_id": "cand-1",
        "adapter": "claude_code",
        "benchmark": "toy",
        "split": "train_probe",
        "tier": "probe",
        "seed": 1,
    }
    payload.update(overrides)
    return EvalRequest(**payload)


@pytest.fixture()
def env(tmp_path: Path):
    """(client, runner, config) with a fresh app per test; TestClient runs the lifespan."""
    config = make_config(tmp_path, make_profile(tmp_path))
    runner = FakeRunner(tmp_path)
    app = build_app(config, runner=runner)
    with TestClient(app) as client:
        yield client, runner, config


def _wait_result(client: TestClient, eval_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.post("/results", json={"eval_id": eval_id}).json()
        if payload["found"]:
            return payload["result"]
        time.sleep(0.05)
    raise AssertionError(f"eval {eval_id} did not settle within {timeout}s")


def _wait_promotion(client: TestClient, promotion_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    payload: dict = {}
    while time.time() < deadline:
        response = client.get(f"/promotion/{promotion_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["state"] not in ("queued", "running"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"promotion {promotion_id} did not settle within {timeout}s: {payload}")


def test_relative_service_write_paths_are_resolved_once_under_workspace(tmp_path: Path) -> None:
    config = make_config(tmp_path, make_profile(tmp_path))
    workspace = tmp_path / "workspace"
    layout = GymLayout.discover(
        repo_root=tmp_path,
        workspace_root=workspace,
        data_root=workspace / "data",
        framework_root=tmp_path / "site-packages",
    )
    config = config.model_copy(
        update={
            "results_root": Path("runtime/results"),
            "ledger_path": Path("runtime/ledger.json"),
            "journal_path": Path("runtime/journal.json"),
            "reports_root": Path("runtime/reports"),
        }
    )
    app = build_app(config, runner=FakeRunner(tmp_path), layout=layout)

    with TestClient(app) as client:
        submitted = client.post("/gym_evaluate", json=make_request().model_dump(mode="json")).json()
        _wait_result(client, submitted["eval_id"])

    eval_dir = workspace / "runtime" / "results" / submitted["eval_id"]
    assert (eval_dir / "result.json").is_file()
    assert (workspace / "runtime" / "ledger.json").is_file()
    assert (workspace / "runtime" / "journal.json").is_file()
    snapshot = json.loads((workspace / "runtime" / "results" / "runtime_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["manifest_schema_version"] == 2
    assert snapshot["environment_digest"].startswith("sha256:")


def _rpc(client: TestClient, method: str, params: dict | None = None, rid: int = 1) -> dict:
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", headers=RPC_HEADERS, json=body).json()


def _handshake(client: TestClient) -> None:
    _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    )
    client.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})


def _call_tool(client: TestClient, name: str, args: dict, rid: int = 3) -> dict:
    return _rpc(client, "tools/call", {"name": name, "arguments": args}, rid=rid)["result"]


def _tool_text(result: dict) -> dict:
    assert result["isError"] is False
    return json.loads(result["content"][0]["text"])


class TestHttpRoutes:
    def test_submit_then_poll_via_results(self, env) -> None:
        client, runner, _ = env
        response = client.post("/gym_evaluate", json=make_request().model_dump())
        assert response.status_code == 200
        submitted = response.json()
        assert submitted["status"] == EvalStatus.QUEUED.value

        result = _wait_result(client, submitted["eval_id"])
        assert result["status"] == EvalStatus.COMPLETED.value
        assert result["metrics"]["rollouts"] == 2
        assert result["metrics"]["reward_mean"] == pytest.approx(0.75)
        assert result["experiment_profile_id"] == "profile-test"
        assert result["cost"]["total_tokens"] == 10

    def test_duplicate_submit_dedupes_to_single_execution(self, env) -> None:
        client, runner, _ = env
        body = make_request().model_dump()
        first = client.post("/gym_evaluate", json=body).json()
        duplicate = client.post("/gym_evaluate", json=body).json()
        assert duplicate["eval_id"] == first["eval_id"]
        assert duplicate["status"] == EvalStatus.QUEUED.value

        result = _wait_result(client, first["eval_id"])
        again = client.post("/gym_evaluate", json=body).json()
        assert again["status"] == EvalStatus.COMPLETED.value
        assert again["eval_id"] == result["eval_id"]
        assert runner.calls == 1

    def test_holdout_rejected_in_band_and_route_absent(self, env) -> None:
        client, _, _ = env
        response = client.post("/gym_evaluate", json=make_request(split="holdout").model_dump())
        assert response.status_code == 200
        rejected = response.json()
        assert rejected["status"] == EvalStatus.REJECTED.value
        assert rejected["error"]["code"] == "HOLDOUT_LOCKED"
        assert not any("holdout" in str(getattr(r, "path", "")) for r in client.app.routes)

    def test_extra_request_field_is_422(self, env) -> None:
        client, _, _ = env
        body = make_request().model_dump()
        body["limit"] = 3  # spec EC-3: the outer loop must not be able to pass this
        assert client.post("/gym_evaluate", json=body).status_code == 422

    def test_tasks_lists_training_split_ids(self, env) -> None:
        client, _, _ = env
        probe = client.post("/tasks", json={"benchmark": "toy", "split": "train_probe"}).json()
        assert len(probe["tasks"]) == 4  # probe_split_size=4 of 10 rows
        assert len(set(probe["tasks"])) == 4
        assert all(task.startswith("toy-") for task in probe["tasks"])
        full = client.post("/tasks", json={"benchmark": "toy", "split": "train_full"}).json()
        assert full["tasks"] == [f"toy-{i}" for i in range(10)]
        # deterministic: repeated listing returns the same probe subset
        assert client.post("/tasks", json={"benchmark": "toy", "split": "train_probe"}).json() == probe

    def test_tasks_rejects_holdout_unknown_benchmark_and_split(self, env) -> None:
        client, _, _ = env
        assert client.post("/tasks", json={"benchmark": "toy", "split": "holdout"}).status_code == 400
        assert client.post("/tasks", json={"benchmark": "nope", "split": "train_probe"}).status_code == 400
        assert client.post("/tasks", json={"benchmark": "toy", "split": "eval"}).status_code == 400

    def test_pinned_task_ids_narrow_input_rows(self, env) -> None:
        client, _, config = env
        probe = client.post("/tasks", json={"benchmark": "toy", "split": "train_probe"}).json()["tasks"]
        pinned = probe[:2]
        submitted = client.post("/gym_evaluate", json=make_request(task_ids=pinned).model_dump()).json()
        result = _wait_result(client, submitted["eval_id"])
        assert result["status"] == EvalStatus.COMPLETED.value
        input_rows = dataset.read_jsonl(Path(config.results_root) / submitted["eval_id"] / "input.jsonl")
        assert [row["task_id"] for row in input_rows] == pinned

    def test_pinned_task_ids_validation_rejections(self, env) -> None:
        client, _, _ = env
        cases = [
            [],  # empty
            ["toy-0", "toy-0"],  # duplicates
            ["toy-0", "toy-999"],  # unknown id
            ["hold-0"],  # id from another split
        ]
        for task_ids in cases:
            response = client.post("/gym_evaluate", json=make_request(task_ids=task_ids).model_dump())
            assert response.status_code == 200
            rejected = response.json()
            assert rejected["status"] == EvalStatus.REJECTED.value
            assert rejected["error"]["code"] == "TASK_SET_INVALID"

    def test_pinned_task_ids_change_idempotency_key(self, env) -> None:
        client, _, _ = env
        probe = client.post("/tasks", json={"benchmark": "toy", "split": "train_probe"}).json()["tasks"]
        first = client.post("/gym_evaluate", json=make_request(task_ids=probe[:2]).model_dump()).json()
        other = client.post("/gym_evaluate", json=make_request(task_ids=probe[2:4]).model_dump()).json()
        assert other["eval_id"] != first["eval_id"]

    def test_pending_counts_queued_work(self, env) -> None:
        client, runner, _ = env
        runner.gate.clear()
        submitted = client.post("/gym_evaluate", json=make_request().model_dump()).json()
        assert client.post("/pending", json={}).json() == {"pending": 1}
        runner.gate.set()
        _wait_result(client, submitted["eval_id"])
        assert client.post("/pending", json={}).json() == {"pending": 0}

    def test_result_replayed_after_restart(self, env) -> None:
        client, _, config = env
        eval_id = client.post("/gym_evaluate", json=make_request().model_dump()).json()["eval_id"]
        _wait_result(client, eval_id)

        restarted = build_app(config, runner=FakeRunner(Path(config.repo_root)))
        with TestClient(restarted) as fresh_client:
            payload = fresh_client.post("/results", json={"eval_id": eval_id}).json()
        assert payload["found"] is True
        assert payload["result"]["status"] == EvalStatus.COMPLETED.value

    def test_feedback_reports_promotion_verdict(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        service = EvaluationService(
            profile=profile,
            runner=FakeRunner(tmp_path),
            ledger=Ledger(config.ledger_path),
            results_root=config.results_root,
            repo_root=tmp_path,
            gym_commit="test-commit",
        )
        holdout_request = make_request(split="holdout", tier="full")
        service.evaluate_holdout(holdout_request)
        PromotionPipeline(service, reports_root=config.reports_root).run(holdout_request)

        app = build_app(config, runner=FakeRunner(tmp_path))
        with TestClient(app) as client:
            found = client.post("/feedback", json={"candidate_id": "cand-1"}).json()
            missing = client.post("/feedback", json={"candidate_id": "ghost"}).json()
        assert found["found"] is True
        assert found["summary"]["holdout"]["holdout_status"] == "completed"
        assert found["summary"]["holdout"]["eligible_for_release"] is True
        assert missing == {"found": False, "summary": None}


class TestPromotionControlPlane:
    DIGEST = "sha256:" + "a" * 64

    @classmethod
    def body(cls, **overrides) -> dict:
        payload = {
            "candidate_id": "cand-1",
            "candidate_digest": cls.DIGEST,
            "adapter": "claude_code",
            "benchmark": "toy",
            "tier": "full",
            "seed": 7,
        }
        payload.update(overrides)
        return payload

    def test_promotion_forces_holdout_and_returns_only_aggregate_report(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        runner = FakeRunner(tmp_path)
        app = build_app(config, runner=runner)
        with TestClient(app) as client:
            response = client.post("/promotion", json=self.body())
            assert response.status_code == 202, response.text
            submitted = response.json()
            assert submitted["state"] == "queued"

            status = _wait_promotion(client, submitted["promotion_id"])
            report = status["report"]
            assert status["state"] == "completed"
            assert report["split"] == "holdout"
            assert report["promotion_id"] == submitted["promotion_id"]
            assert report["candidate_digest"] == self.DIGEST
            assert report["metrics"]["rollouts"] == 2
            assert "task_ids" not in report
            assert "per_rollout_ref" not in report

            rows = dataset.read_jsonl(Path(config.results_root) / report["eval_id"] / "input.jsonl")
            assert [row["task_id"] for row in rows] == ["hold-0", "hold-1", "hold-2"]

            feedback = client.post("/feedback", json={"candidate_id": "cand-1"}).json()
            assert feedback["found"] is True
            assert runner.calls == 1  # /feedback reads the report; it never reruns holdout

    def test_promotion_runs_complete_holdout_without_limit(self, tmp_path: Path) -> None:
        """F-01 end-to-end: even a budget.max_rollouts cap cannot truncate the holdout argv."""
        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        app = build_app(config, runner=FakeRunner(tmp_path))
        with TestClient(app) as client:
            response = client.post("/promotion", json=self.body(budget={"max_rollouts": 1}))
            assert response.status_code == 202, response.text
            status = _wait_promotion(client, response.json()["promotion_id"])
            assert status["state"] == "completed"
            repro = json.loads(
                (Path(config.results_root) / status["report"]["eval_id"] / "repro.json").read_text(encoding="utf-8")
            )
            assert "--limit" not in repro["argv"]

    def test_promotion_is_one_shot_per_bundle_and_environment(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        runner = FakeRunner(tmp_path)
        app = build_app(config, runner=runner)
        with TestClient(app) as client:
            first = client.post("/promotion", json=self.body())
            assert first.status_code == 202
            _wait_promotion(client, first.json()["promotion_id"])

            duplicate = client.post("/promotion", json=self.body())
            assert duplicate.status_code == 200
            assert duplicate.json()["reused"] is True
            assert duplicate.json()["promotion_id"] == first.json()["promotion_id"]
            assert runner.calls == 1

            # A new seed is NOT a new experiment identity: the holdout split is one-shot
            # per candidate content + environment.
            new_seed = client.post("/promotion", json=self.body(seed=8))
            assert new_seed.status_code == 200
            assert new_seed.json()["reused"] is True
            assert new_seed.json()["promotion_id"] == first.json()["promotion_id"]
            assert runner.calls == 1

            other_bundle = client.post(
                "/promotion", json=self.body(candidate_digest="sha256:" + "b" * 64)
            )
            assert other_bundle.status_code == 202
            assert other_bundle.json()["promotion_id"] != first.json()["promotion_id"]

    def test_concurrent_duplicate_promotion_is_rejected(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        runner = FakeRunner(tmp_path)
        runner.gate.clear()
        app = build_app(config, runner=runner)
        with TestClient(app) as client:
            first = client.post("/promotion", json=self.body())
            assert first.status_code == 202
            duplicate = client.post("/promotion", json=self.body())
            assert duplicate.status_code == 409
            assert duplicate.json()["detail"]["code"] == "PROMOTION_IN_PROGRESS"
            runner.gate.set()
            _wait_promotion(client, first.json()["promotion_id"])

    def test_promotion_request_cannot_express_split_or_task_ids(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        app = build_app(config, runner=FakeRunner(tmp_path))
        with TestClient(app) as client:
            assert client.post("/promotion", json=self.body(split="holdout")).status_code == 422
            assert client.post("/promotion", json=self.body(task_ids=["hold-0"])).status_code == 422
            # F-01: a capped tier can never truncate the holdout.
            assert client.post("/promotion", json=self.body(tier="probe")).status_code == 422
            assert client.post("/promotion", json=self.body(tier="screen")).status_code == 422
            body = self.body()
            body.pop("candidate_digest")
            assert client.post("/promotion", json=body).status_code == 422

    def test_promotion_preflight_rejects_missing_holdout_dataset(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=False)
        config = make_config(tmp_path, profile)
        app = build_app(config, runner=FakeRunner(tmp_path))
        with TestClient(app) as client:
            response = client.post("/promotion", json=self.body())
            assert response.status_code == 400
            assert response.json()["detail"]["code"] == "INVALID_SPLIT"

    def test_promotion_loopback_guard_and_unknown_id(self, tmp_path: Path) -> None:
        assert _is_loopback_client("127.0.0.1")
        assert _is_loopback_client("::1")
        assert _is_loopback_client("testclient")
        assert not _is_loopback_client("10.0.0.8")

        profile = make_profile(tmp_path, with_holdout=True)
        config = make_config(tmp_path, profile)
        app = build_app(config, runner=FakeRunner(tmp_path))
        with TestClient(app) as client:
            assert client.get("/promotion/promo-0000000000000000").status_code == 404
            assert client.get("/promotion/not-a-promotion-id").status_code == 404


class TestMcpSurface:
    def test_tools_list_matches_contract_verbatim(self, env) -> None:
        client, _, _ = env
        _handshake(client)
        tools = {t["name"]: t for t in _rpc(client, "tools/list", {}, rid=2)["result"]["tools"]}
        assert set(tools) == {GYM_EVALUATE_TOOL_NAME, RESULT_TOOL_NAME, FEEDBACK_TOOL_NAME}
        assert tools[GYM_EVALUATE_TOOL_NAME]["inputSchema"] == EVAL_TOOL_PARAMETERS
        assert tools[GYM_EVALUATE_TOOL_NAME]["description"] == GYM_EVALUATE_TOOL_DESCRIPTION

    def test_mcp_evaluate_then_result_roundtrip(self, env) -> None:
        client, _, _ = env
        _handshake(client)
        submitted = _tool_text(_call_tool(client, GYM_EVALUATE_TOOL_NAME, make_request().model_dump()))
        assert submitted["status"] == EvalStatus.QUEUED.value

        payload = None
        deadline = time.time() + 10
        while time.time() < deadline:
            payload = _tool_text(_call_tool(client, RESULT_TOOL_NAME, {"eval_id": submitted["eval_id"]}))
            if payload["found"]:
                break
            time.sleep(0.05)
        assert payload is not None and payload["result"]["status"] == EvalStatus.COMPLETED.value

    def test_mcp_contract_violation_is_error_result(self, env) -> None:
        client, _, _ = env
        _handshake(client)
        body = make_request().model_dump()
        body["limit"] = 3
        result = _call_tool(client, GYM_EVALUATE_TOOL_NAME, body)
        assert result["isError"] is True

    def test_mcp_rejects_host_only_candidate_digest(self, env) -> None:
        """F-04: the MCP data plane cannot forge the host-supplied candidate identity."""
        client, runner, _ = env
        _handshake(client)
        body = make_request().model_dump()
        body["candidate_digest"] = "sha256:" + "a" * 64
        result = _call_tool(client, GYM_EVALUATE_TOOL_NAME, body)
        assert result["isError"] is True
        assert "candidate_digest" in result["content"][0]["text"]
        assert runner.calls == 0

    def test_mcp_unknown_tool_is_error_result(self, env) -> None:
        client, _, _ = env
        _handshake(client)
        result = _call_tool(client, "nope", {})
        assert result["isError"] is True
        assert "nope" in result["content"][0]["text"]

    def test_mcp_disabled_returns_404(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, make_profile(tmp_path))
        app = build_app(config.model_copy(update={"expose_tools_over_mcp": False}), runner=FakeRunner(tmp_path))
        with TestClient(app) as client:
            assert client.post("/mcp", headers=RPC_HEADERS, json={}).status_code == 404
