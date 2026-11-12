# ARC-AGI resources server

Few-shot grid transformation puzzles (Abstraction and Reasoning Corpus).
Each task shows example input/output grid pairs plus a test input grid; the
model must apply the induced transformation and return the output grid.

Verification is a deterministic exact match against the expected output grid
(cell-for-cell, dimensions included), following ARC Prize scoring. The
predicted grid is extracted from the response by parsing JSON arrays of
arrays (fenced code blocks preferred, then the rightmost parsable candidate
in the raw text), with lenient handling of string-digit cells.

Task fields carried on the wire (see `task_data.py`): `question`
(pre-rendered examples + test input), `train_pairs`, `test_input`, and
`expected_output_grid`.
