"""Idempotent prepare for the rsi-traj-syn benchmark (ADF RSI trajectory synthesis).

Reads the versioned data file written by the ADF publisher and materializes
the canonical Gym JSONL (question / expected_answer / reference_solution).
No network access — provenance (lessons / cluster labels / fingerprints)
lives in the versioned file's `info` field and the ADF-side artifact manifest.
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "rsi-traj-syn_v1.jsonl"
OUTPUT_FPATH = DATA_DIR / "rsi-traj-syn_benchmark.jsonl"
_FIELDS = ['question', 'expected_answer', 'reference_solution']


def prepare() -> Path:
    """Materialize canonical rows from the versioned data file (idempotent)."""
    rows = [
        json.loads(line)
        for line in SOURCE_FPATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUTPUT_FPATH.write_text(
        "".join(
            json.dumps({f: row[f] for f in _FIELDS}, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
