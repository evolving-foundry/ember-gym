# Direct RSI and long-horizon Agent benchmarks

These benchmarks test improvement over experience, persistent cross-session learning, or an agent
that changes its own code/model. They do not fit Gym's one-request/one-rollout `gym eval run` contract.
Use the upstream evaluator and record its raw outputs alongside this repository's experiment
manifest. `sources.lock.json` pins the sources inspected for this integration plan.

The commands below are upstream commands, not local Gym wrappers. Run them in Linux/WSL2 with the
source checkout at the commit in `sources.lock.json` (check out that hash after cloning). Keep each
benchmark's native score and report it separately; do not combine them into the Gym task score.

## Priority and integration choice

| Priority | Benchmark | What it measures | Project integration |
|---|---|---|---|
| P0 | [EvoAgentBench](https://github.com/EverMind-AI/EvoAgentBench) | Reusable procedural knowledge transferred from train trajectories to held-out tasks across four domains. | Run its unified upstream runner. Keep its official 528/267 ID split. |
| P0 | [PAST-Bench](https://github.com/Gen-Verse/PAST-Bench) | Matched persistence-on/off performance over ordered fresh-session episode sequences, plus artifact/trace evidence. | Run its sequence runner; capture per-family sequence summaries and comparison deltas. |
| P0 | [RSI-Exam](https://github.com/aiming-lab/RSI-Exam) | Long-horizon experiments on visible data, then one clean replay against sealed data. | Use Harbor and the public tasks; full 88-task results require benchmark access. |
| P1 | [AI4AI-Bench](https://github.com/Einsia/AI4AI-Bench) | Agent modification of ML training algorithms under resource ceilings, then isolated formal training and validation. | Use its orchestrator on Linux amd64 with a supported NVIDIA GPU. Local scores are not official blind-eval results. |
| P1 | [BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard.html) | Function/tool calling, multi-turn interaction, and agentic tool-use categories. | Treat each BFCL release and evaluator commit as a versioned tool-use track; don't mix periodic live data with offline cases. |
| P2 | [AgentBench FC](https://github.com/THUDM/AgentBench) | Multi-turn interaction with containerized ALFWorld, DBBench, Knowledge Graph, OS, and WebShop workers. | Use the FC release as its own benchmark; results are not comparable to AgentBench v0.1/v0.2 eight-task composite. |
| P2 | [OSWorld-Science](https://huggingface.co/datasets/SciAILab/osworld-science) | Desktop science tasks scored on machine-checkable scientific deliverables. | Separate OSWorld-derived runner; HF requires accepting access conditions. Dataset publishes answer keys, so it is not a blind holdout. |
| Watch | [Knows](https://github.com/ServiceNow/BrowserGym/pull/397) | Long-horizon Google Workspace workflows with checkpoints. | Wait for a released and versioned integration plus workspace credentials and evaluator details. |

P0/P1 denotes the order for this project's RSI evidence: direct transfer/sequence evaluation first,
then expensive research-algorithm tracks. P2 is useful capability coverage but older or less aligned
with recursive improvement. “Watch” is not counted as an available runnable integration.

## EvoAgentBench

The upstream repo supplies a unified four-domain runner and the official 528 train / 267 test task-ID
splits. Payloads, retrieval indexes, containers, and keys are external assets. Prepare only the domain
being evaluated and start with a small split before any full run.

```bash
git clone https://github.com/EverMind-AI/EvoAgentBench.git vendor/EvoAgentBench
cd vendor/EvoAgentBench/benchmark
conda env create -f environment.yml
conda activate evoagentbench
pip install -r requirements.txt
pip install -r requirements-ir.txt       # information_retrieval, optional
pip install -r requirements-code.txt     # code_implementation, optional
pip install -r requirements-swe.txt      # software_engineering, optional
pip install -r requirements-kw.txt       # knowledge_work, optional

# Set up the selected agent and benchmark data following that domain's guide.
python src/run.py --domain information_retrieval --split 2 --job ir-smoke --live
python src/run.py --domain information_retrieval --split train --parallel 4 --job ir-train
python src/run.py --domain information_retrieval --split test --parallel 4 --job ir-baseline
```

For an evolution-method result, use the upstream extraction/evaluation workflow after the train run.
Do not replace its official test IDs with project-created Gym splits. Record domain, upstream commit,
agent and model config, task payload/index hashes, test job ID, and all train/evolution artifacts.

## PAST-Bench

PAST-Bench includes 26 families / 204 episodes in four abilities. Each episode begins as a fresh
session; sequence ordering and the persistence-disabled paired control are part of the score.

```bash
git clone https://github.com/Gen-Verse/PAST-Bench.git vendor/PAST-Bench
cd vendor/PAST-Bench
uv venv --python 3.11 --seed
source .venv/bin/activate
uv pip install -e ".[mock,sandbox,dev]"
uv pip install -e agents/hermes-agent
uv pip install -e agents/hermes-plus
past-bench build-image --kind sandbox

HERMES_PLUS_MECHANISMS=all past-bench evolve \
  --family memory_ability/SM01_preference_adoption \
  --agent hermes-plus --agent-profile minimax --runtime local \
  --sandbox --sandbox-tools --compare-no-persistence \
  --judge-model MiniMax-M2.7 \
  --trace-dir traces/agentgym/smoke/SM01_preference_adoption
```

Use the upstream `scripts/run_hermes_plus_mechanisms_full.sh` or family list for the full run. Keep
`sequence_results.json`, `sequence_summary.json`, and `sequence_comparison.json` per family. Do not
flatten episode rows into independent Gym trials: that loses cross-session state and the paired delta.

## RSI-Exam

The benchmark's public dataset contains 35 runnable tasks; the full suite has 88 tasks and requires
contacting the maintainers for the private 53. Each public task allows up to 12 hours of exploration,
then the submitted artifact is replayed once in a fresh environment against sealed data.

```bash
git clone https://github.com/aiming-lab/RSI-Exam.git vendor/RSI-Exam
cd vendor/RSI-Exam
uv tool install harbor
hf download RSI-Exam/RSI-Exam --repo-type dataset --local-dir tasks/

export ARB_PROGRAM="$PWD/infra/prompts/autoresearch.md"
export ARB_BUDGET_PY="$PWD/infra/prompts/budget.py"
export ARB_AGENT_TIMEOUT_SEC=43200
export ARB_OUTPUT_TOKEN_LIMIT=500000

# Fill the matching task-side/API credentials as documented upstream, then run one task.
harbor run -p tasks/<public-task> -a codex -m <model> -y -n 1 \
  --ak prompt_template_path="$PWD/infra/prompts/autoresearch.j2" \
  --extra-docker-compose "$PWD/infra/prompts/mount.yaml"
harbor view jobs
```

Use the exact upstream harness flags for the chosen agent and model (network allow-lists and tool
restrictions are part of the protocol). Preserve Harbor job metadata and submitted artifact hashes.
Never copy private tasks or sealed answers into this repository.

## AI4AI-Bench

AI4AI-Bench currently defines ten algorithmic-design tasks. Its reference run targets Linux amd64,
Docker, NVIDIA Container Toolkit, and one B300; other GPUs support local development only when the
task fits. The benchmark distinguishes exploration from formal retraining and supports up to three
checkpoints.

```bash
git clone https://github.com/Einsia/AI4AI-Bench.git vendor/AI4AI-Bench
cd vendor/AI4AI-Bench
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[assets]'
export AI4AI_ASSET_STORE=/data/ai4ai/assets
python tools/prepare_assets.py --task <task-id> \
  --assets "$AI4AI_ASSET_STORE/<task-id>" --execute
python tools/verify_assets.py --task <task-id> \
  --assets "$AI4AI_ASSET_STORE/<task-id>" --hash
bash tools/smoke.sh --root /data/ai4ai/smoke --gpu 0
python tools/check_setup.py --task <task-id> \
  --assets "$AI4AI_ASSET_STORE/<task-id>" \
  --root /data/ai4ai/runs --gpu 0 --agent codex --mode local
bash orchestrator/trial.sh <run-id> --task tasks/<task-id> \
  --assets "$AI4AI_ASSET_STORE/<task-id>" \
  --root /data/ai4ai/runs --gpu 0 --agent codex \
  --model <model> --reasoning-effort high
```

The upstream docs currently describe self-hosted final evaluation, not a blind evaluation service.
Label local results accordingly; save asset manifest/hash, task TOML, GPU/model identities, candidate
patch, checkpoints, resource logs, and evaluation receipts.

## BFCL V4 and AgentBench FC

BFCL's official leaderboard changes over time. The cited leaderboard states its reproduced models
use evaluator commit `f7cf735` and package `bfcl-eval==2025.12.17`; pin the release under test and
run the upstream evaluation CLI for supported categories. Keep live/API categories separate from
offline deterministic categories and report per-category scores.

AgentBench's current `main` is the newer Function Calling release, with Docker Compose workers for
ALFWorld, DBBench, Knowledge Graph, OS Interaction, and WebShop. It requires extra images and, for
Knowledge Graph, a Freebase database. The original eight-task benchmark remains in historical tags;
do not report an FC result under the original composite name.

## Connecting upstream results to RSI Gym

Use this directory as the protocol registry. Keep upstream run IDs/artifact paths in the experiment
record and attach a short result summary to the RSI run lineage. The ordinary Gym profile can still
screen candidates on OSWorld, SWE-bench, τ²/τ³, BrowserGym, AssistantBench, and Terminal-Bench; final
RSI claims should include the matching native sequence/transfer/research protocol when applicable.
An adapter into `eval_service.runner.GymEvalRunner` is intentionally not defined for these suites:
the current runner invokes one `gym eval run --input JSONL` subprocess and cannot preserve their native
paired state, sealed replay, or fresh-training phases.

## Official source revision locks

`sources.lock.json` records the Git commit IDs inspected on 2026-10-05. Re-check licenses, task
versions, and upstream command syntax when intentionally upgrading any pin.
