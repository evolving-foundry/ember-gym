# GAIA text validation subset

This Gym integration prepares GAIA's public validation questions that have no file attachments,
uses deterministic GAIA quasi-exact answer scoring, and keeps the official test split out of the
repository workflow. It is a project-specific validation subset; scores are not an official GAIA
leaderboard submission.

GAIA data access is gated by the dataset host. Request access and authenticate with Hugging Face
before preparing:

```powershell
python scripts/run_component.py benchmarks/gaia/prepare.py --levels 1 2 3
```

The generated JSONL is ignored by git. To create the RSI search/holdout files, run:

```powershell
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/gaia/data/gaia_text_validation.jsonl `
  --train benchmarks/gaia/data/rsi/gaia_train.jsonl `
  --holdout benchmarks/gaia/data/rsi/gaia_holdout.jsonl `
  --id-field task_id
```

Preparation writes complete Gym request rows, including the `responses_create_params` model input
and the grading-only `expected_answer`; `eval_service` sends the former to the agent and the latter
to the deterministic verifier. If Hugging Face rejects access, the script leaves the prior dataset
file intact instead of replacing it with an empty partial file.

Start `eval_service` with `eval_service/profiles/rsi_v1_gaia.yaml`. This first integration skips
image, audio, PDF, spreadsheet, and other attachment tasks. Adding them needs a versioned attachment
materializer and an agent runtime that can read those artifacts.
