#!/usr/bin/env python3
"""Generate the deterministic ARC-AGI example used by ``arc_agi_simple_agent``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPONENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = COMPONENT_DIR / "data" / "example.jsonl"

TRAIN_PAIRS = [
    {
        "input": [[8, 6], [6, 4]],
        "output": [
            [8, 6, 8, 6, 8, 6],
            [6, 4, 6, 4, 6, 4],
            [6, 8, 6, 8, 6, 8],
            [4, 6, 4, 6, 4, 6],
            [8, 6, 8, 6, 8, 6],
            [6, 4, 6, 4, 6, 4],
        ],
    },
    {
        "input": [[1, 2], [2, 1]],
        "output": [
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
        ],
    },
]
TEST_INPUT = [[6, 8], [4, 6]]
EXPECTED_OUTPUT_GRID = [
    [6, 8, 6, 8, 6, 8],
    [4, 6, 4, 6, 4, 6],
    [6, 8, 6, 8, 6, 8],
    [4, 6, 4, 6, 4, 6],
]


def _grid_json(grid: list[list[int]]) -> str:
    return json.dumps(grid, separators=(", ", ": "))


def render_question(train_pairs: list[dict[str, Any]], test_input: list[list[int]]) -> str:
    lines: list[str] = []
    for index, pair in enumerate(train_pairs, start=1):
        lines.extend(
            [
                f"Example {index}:",
                f"Input: {_grid_json(pair['input'])}",
                f"Output: {_grid_json(pair['output'])}",
                "",
            ]
        )
    lines.append(f"Test input: {_grid_json(test_input)}")
    return "\n".join(lines)


def example_row() -> dict[str, Any]:
    return {
        "question": render_question(TRAIN_PAIRS, TEST_INPUT),
        "train_pairs": TRAIN_PAIRS,
        "test_input": TEST_INPUT,
        "expected_output_grid": EXPECTED_OUTPUT_GRID,
    }


def write_example(output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(example_row(), ensure_ascii=False))
        handle.write("\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    output = write_example(arguments.output)
    print(f"Wrote ARC-AGI example fixture: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
