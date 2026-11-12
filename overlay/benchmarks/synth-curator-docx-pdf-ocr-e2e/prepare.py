"""Materialize the synth-curator-docx-pdf-ocr-e2e synth_env benchmark from its package zip."""

import argparse
import json
import sys
import zipfile
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
PACKAGE_ZIP = BENCHMARK_DIR / "assets" / "gym-env-package-v1-ee3992a5604b89eb.zip"
OUTPUT_JSONL = BENCHMARK_DIR / "data" / "synth-curator-docx-pdf-ocr-e2e.jsonl"
REQUIRED_FIELDS = ["task_id", "env_release", "goal", "success_check", "max_steps", "seed"]
SUPPORTED_VERIFIERS = ("state_assert", "artifact_compare")
FORBIDDEN_KEYS = ("gold", "gold_solution", "negative_solution")


def _reject(index, message):
    raise ValueError("row {}: {}".format(index, message))


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _reject_private(value, path="task"):
    if isinstance(value, dict):
        leaked = sorted(set(FORBIDDEN_KEYS) & set(value))
        if leaked:
            raise ValueError("{} leaks private answer fields {}".format(path, leaked))
        for key, item in value.items():
            _reject_private(item, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private(item, "{}[{}]".format(path, index))


def _validate_payload(kind, payload):
    if kind == "state_assert":
        if set(payload) != {"checks"}:
            raise ValueError("state_assert payload must contain only checks")
        checks = payload.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError("state_assert checks must be a non-empty list")
        for index, check in enumerate(checks):
            if not isinstance(check, dict) or set(check) != {"collection", "match", "expect"}:
                raise ValueError("state_assert check {} fields are invalid".format(index))
            if not _non_empty_string(check["collection"]):
                raise ValueError("state_assert check {} collection is invalid".format(index))
            if not isinstance(check["match"], dict) or not check["match"]:
                raise ValueError("state_assert check {} match is invalid".format(index))
            if not isinstance(check["expect"], dict) or not check["expect"]:
                raise ValueError("state_assert check {} expect is invalid".format(index))
        return
    if kind == "artifact_compare":
        if set(payload) != {"expect_files"}:
            raise ValueError("artifact_compare payload must contain only expect_files")
        expected = payload.get("expect_files")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("artifact_compare expect_files must be a non-empty object")
        for key, value in expected.items():
            if not _non_empty_string(key) or not isinstance(value, str):
                raise ValueError("artifact_compare expect_files must map paths to strings")


def _validate(rows):
    if not rows:
        raise ValueError("package contains no tasks")
    seen = set()
    pins = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _reject(index, "task must be an object")
        try:
            _reject_private(row, "row {}".format(index))
        except ValueError as error:
            _reject(index, str(error))
        if set(row) != set(REQUIRED_FIELDS):
            _reject(index, "fields must be exactly {}".format(REQUIRED_FIELDS))
        task_id = row["task_id"]
        if not _non_empty_string(task_id):
            _reject(index, "task_id must be a non-empty string")
        if task_id in seen:
            _reject(index, "duplicate task_id: {}".format(task_id))
        seen.add(task_id)
        if not _non_empty_string(row["goal"]):
            _reject(index, "goal must be a non-empty string")
        release = row["env_release"]
        if not isinstance(release, dict) or set(release) != {"ref_id", "version", "fingerprint"}:
            _reject(index, "env_release fields are invalid")
        if not _non_empty_string(release["ref_id"]):
            _reject(index, "env_release.ref_id is invalid")
        if not isinstance(release["version"], int) or isinstance(release["version"], bool) or release["version"] < 1:
            _reject(index, "env_release.version must be a positive integer")
        fingerprint = release["fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 16
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            _reject(index, "env_release.fingerprint must be 16 lowercase hex chars")
        pins.add((release["ref_id"], release["version"], fingerprint))
        success_check = row["success_check"]
        if not isinstance(success_check, dict) or set(success_check) != {"verifier_spec"}:
            _reject(index, "success_check fields are invalid")
        verifier = success_check["verifier_spec"]
        if not isinstance(verifier, dict) or set(verifier) != {"kind", "payload"}:
            _reject(index, "verifier fields are invalid")
        if verifier["kind"] not in SUPPORTED_VERIFIERS:
            _reject(index, "unsupported verifier kind: {}".format(verifier["kind"]))
        if not isinstance(verifier["payload"], dict) or not verifier["payload"]:
            _reject(index, "verifier payload must be a non-empty object")
        try:
            _validate_payload(verifier["kind"], verifier["payload"])
        except ValueError as error:
            _reject(index, str(error))
        for field in ("max_steps", "seed"):
            if not isinstance(row[field], int) or isinstance(row[field], bool):
                _reject(index, "{} must be an integer".format(field))
        if row["max_steps"] < 1:
            _reject(index, "max_steps must be >= 1")
    if len(pins) != 1:
        raise ValueError("package tasks pin {} different releases".format(len(pins)))
    return rows, next(iter(pins))


def prepare(package_zip: Path = PACKAGE_ZIP, output_jsonl: Path = OUTPUT_JSONL) -> Path:
    with zipfile.ZipFile(package_zip) as archive:
        raw = archive.read("tasks.jsonl").decode("utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows, pin = _validate(rows)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print("wrote {} ({} tasks, pin {})".format(output_jsonl, len(rows), pin))
    return output_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-zip", type=Path, default=PACKAGE_ZIP)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL)
    args = parser.parse_args()
    try:
        prepare(args.package_zip, args.output)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        print("prepare failed: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
