from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "overlay" / "resources_servers" / "arc_agi" / "prepare_example.py"
TASK_DATA_PATH = REPO_ROOT / "overlay" / "resources_servers" / "arc_agi" / "task_data.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_is_deterministic_and_matches_arc_agi_schema(tmp_path: Path) -> None:
    module = _load_module("arc_agi_prepare_example", MODULE_PATH)
    task_data = _load_module("arc_agi_task_data", TASK_DATA_PATH)
    output = tmp_path / "example.jsonl"

    module.write_example(output)
    first = output.read_bytes()
    module.write_example(output)

    assert output.read_bytes() == first
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1

    row = rows[0]
    assert list(row) == ["question", "train_pairs", "test_input", "expected_output_grid"]
    assert row["question"] == module.render_question(module.TRAIN_PAIRS, module.TEST_INPUT)
    assert row["train_pairs"] == module.TRAIN_PAIRS
    assert row["test_input"] == module.TEST_INPUT
    assert row["expected_output_grid"] == module.EXPECTED_OUTPUT_GRID
    task_data.TaskData.model_validate(row)
