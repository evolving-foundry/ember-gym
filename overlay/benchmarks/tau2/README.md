# Tau2 / Tau3 Banking Benchmarks

This benchmark directory uses one public prepare entrypoint:

```bash
python scripts/run_component.py benchmarks/tau2/prepare.py
```

With no arguments it prepares the original Tau2 benchmark domains: `airline`,
`retail`, and `telecom`.

Tau3 `banking_knowledge` is opt-in through prepare arguments:

```bash
python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --retrieval-config bm25_grep
python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --retrieval-config terminal_use
python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --retrieval-config alltools
python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --all
```

Gym config files live under the `benchmarks/tau2/configs/` component path (source:
`overlay/benchmarks/tau2/configs/`):

- `tau2.yaml`: original Tau2 `airline`, `retail`, and `telecom`.
- `banking_bm25_grep_artificial_analysis.yaml`: Tau3 `banking_knowledge` with
  BM25 lexical retrieval plus grep, a GPT-5.4 Mini user simulator, and five
  repeats following the Artificial Analysis run shape.
- `banking_terminal_use.yaml`: Tau3 `banking_knowledge` with `terminal_use`.
- `banking_alltools.yaml`: Tau3 `banking_knowledge` with `alltools`.

The banking configs keep Gym's existing no-argument `prepare()` contract by
pointing at small wrapper modules under `prepare_utils/`:

```yaml
prepare_script: benchmarks/tau2/prepare_utils/banking_terminal_use.py
```

Prepare implementation details are under `benchmarks/tau2/prepare_utils/`.
The generated JSONL files are written to `benchmarks/tau2/data/` and are
ignored by git.

## Pinned Tau Data Source

Benchmark preparation clones a fixed Tau data-generation commit:

```text
https://github.com/bxyu-nvidia/tau2-bench@ce4013b0afe03c873488878b72851414f92f458b
```

That commit owns `dump_nemo_gym_data.py`; preparation runs it in the pinned
Tau environment and then reads the generated `nemo_gym_data` JSON files. On
Windows, only the temporary filenames are changed to stable numeric names
(task IDs remain unchanged in the JSON rows). Every prepared row also receives
a globally unique `task_id` composed from its domain and upstream task ID.
Override the source for local testing
with:

```bash
NEMO_GYM_TAU2_BENCH_DATA_REPO_URL=/path/to/tau2-bench \
NEMO_GYM_TAU2_BENCH_DATA_REF=ce4013b0afe03c873488878b72851414f92f458b \
python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --retrieval-config terminal_use
```

The pinned data snapshot keeps the original `bxyu/nemo_gym_data` interface: a
`dump_nemo_gym_data.py` script emits `nemo_gym_data/<dataset>/*.json` rows with
Tau config, task, system prompt, and Responses-style tool schemas. Internally it
uses Tau's build layer directly instead of patching `run_single_task`, which
keeps the branch smaller while preserving the generated row contract.

## Evaluation Semantics

`banking_bm25_grep_artificial_analysis.yaml` follows the
[Artificial Analysis Tau3-Banking methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking#tau3-banking)
run shape: all 97 tasks, five repeats, `bm25_grep` retrieval, a GPT-5.4 Mini
user simulator with medium reasoning, at most 200 steps, and at most 10
tool-execution errors per task repeat. The error limit comes from the pinned Tau
task rows. The user simulator uses the pinned OpenAI API snapshot
`gpt-5.4-mini-2026-03-17`.

Prepared Gym rows remove `NL_ASSERTION` from each task's `reward_basis`, so Gym
does not call the external LLM judge. In the current `banking_knowledge` task
set, this changes only `task_102`; the other 96 banking tasks do not use an NL
assertion reward. Banking scores produced through this Gym integration are
therefore no-judge scores and are not strictly identical to Artificial
Analysis, which uses GPT-5.4 Mini as the NL-assertion judge.

## Runtime Checks

`bm25_grep` needs no retrieval API key or sandbox tooling. `terminal_use` and
`alltools` need sandbox tooling at rollout time, and `alltools` also needs
`OPENAI_API_KEY` for dense retrieval. Check a retrieval config with:

```bash
python scripts/run_component.py benchmarks/tau2/prepare_utils/runtime.py bm25_grep
python scripts/run_component.py benchmarks/tau2/prepare_utils/runtime.py terminal_use
python scripts/run_component.py benchmarks/tau2/prepare_utils/runtime.py alltools
```

## RSI project splits

The RSI profile uses `configs/rsi_tau2.yaml` and
`configs/rsi_banking_bm25_grep.yaml`. These register a single Gym agent key,
`rsi_tau_agent`, for both environments. Its policy turns go to the RSI
candidate's OpenAI-compatible Chat Completions endpoint via
`rsi_agent_endpoint`; simulated-user turns go to the separate
`rsi_tau_user_model`. The RSI endpoint must implement conversation history and
tool calls, and the RSI materializer must activate the requested candidate
before each evaluation. A plain model endpoint measures that model rather than
the RSI agent's full behavior.

Set `RSI_AGENT_CHAT_BASE_URL`, `RSI_AGENT_CHAT_API_KEY`, and
`RSI_AGENT_CHAT_MODEL` for the candidate endpoint. Set
`TAU_USER_BASE_URL`, `TAU_USER_API_KEY`, and `TAU_USER_MODEL_NAME` for the
simulated user. `TAU_USER_THINKING_TYPE` is optional. These values do not configure the
candidate. A different simulator changes the evaluation method and its
manifest digest. The shipped endpoint configs use the Gym `openai_model` transport,
which accepts compatible Chat Completions services. Another Gym model-server
implementation can be registered under `rsi_agent_endpoint` by replacing the
adapter's `config_paths` in the pinned RSI profile.

For the RSI τ² environment, start Gym with both the candidate transport and
the benchmark composition, for example:

```bash
gym env start '+config_paths=[benchmarks/tau2/configs/rsi_tau2.yaml,responses_api_models/openai_model/configs/rsi_agent_chat_endpoint.yaml]'
```

For Banking, replace `rsi_tau2.yaml` with `rsi_banking_bm25_grep.yaml`.
The original benchmark YAMLs remain reference-agent configurations and still
require their own OpenAI credentials. Changing the simulated-user model or
provider changes the evaluation harness and its manifest digest; do not
compare scores across those changes as one experiment.

To use a MiniMax settings file without copying its key into the repository or
command line, set `MINIMAX_SETTINGS_PATH` to its absolute path and launch each
Gym process through `run_with_minimax.py`:

```bash
python scripts/run_component.py benchmarks/tau2/run_with_minimax.py \
  -- \
  gym env start '+config_paths=[benchmarks/tau2/configs/rsi_tau2.yaml,responses_api_models/openai_model/configs/rsi_agent_chat_endpoint.yaml]'
```

Use the same wrapper when launching `eval_service` so its Gym subprocess
inherits the simulator settings. The script derives the OpenAI-compatible
`/v1` URL and model from the MiniMax Claude settings; it passes the key only
in the child environment. `RSI_AGENT_CHAT_*` still identifies the separate
candidate Agent endpoint.

Prepare the source rows first, then create stable search/holdout files. The split is by task ID and
is not an official τ benchmark split:

```powershell
python scripts/run_component.py benchmarks/tau2/prepare.py
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/tau2/data/tau2_benchmark.jsonl `
  --train benchmarks/tau2/data/rsi/tau2_train.jsonl `
  --holdout benchmarks/tau2/data/rsi/tau2_holdout.jsonl `
  --id-field task_id

python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --retrieval-config bm25_grep
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/tau2/data/tau2_banking_knowledge_bm25_grep.jsonl `
  --train benchmarks/tau2/data/rsi/tau3_banking_train.jsonl `
  --holdout benchmarks/tau2/data/rsi/tau3_banking_holdout.jsonl `
  --id-field task_id
```

These paths and field names match `eval_service/profiles/rsi_v1_tau2.yaml`. For a valid comparison,
keep the source commit, τ package and evaluator version, user-simulator model, domain, retrieval
configuration, judge behavior, and repeat count fixed across candidates. In particular, the
`bm25_grep` Gym config strips `NL_ASSERTION`, so its score is not an Artificial Analysis leaderboard
reproduction.
