#!/usr/bin/env bash
set -euo pipefail

# Rebuild the benchmark artifacts that are intentionally not tracked by Git.
# Run with `--dry-run` to inspect the upstream commands without downloading data.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

export AGENT_GYM_REPO="${AGENT_GYM_REPO:-${repo_root}}"
export AGENT_GYM_OVERLAY="${AGENT_GYM_OVERLAY:-${repo_root}/overlay}"
export PYTHONPATH="${AGENT_GYM_OVERLAY}:${AGENT_GYM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

gym_bin="${GYM_BIN:-agent-gym}"
output_root="${AGENT_GYM_PREPARE_OUTPUT:-${repo_root}/.tmp/data-prep}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    gym_path="${gym_bin}"
    if [[ "${gym_path}" != */* ]]; then
        gym_path="$(command -v "${gym_bin}" 2>/dev/null || true)"
    fi
    if [[ -n "${gym_path}" && -x "$(dirname "${gym_path}")/python" ]]; then
        PYTHON_BIN="$(dirname "${gym_path}")/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
fi

all_benchmarks=(aegis_v4_safety arc-agi arc-agi-v2 arc-agi-v3 gaia gsm8k)
renderable_benchmarks=(arc-agi arc-agi-v2 gsm8k)
collateable_benchmarks=(aegis_v4_safety arc-agi arc-agi-v2 gsm8k)
rsi_benchmarks=(arc-agi gaia)
dry_run=0

usage() {
    cat <<'EOF'
Usage: bash scripts/prepare_data.sh [--dry-run] <command> [name ...]

Commands:
  prepare [name ...]  Run `gym eval prepare` for the named benchmark(s).
  collate [name ...]  Write metrics + *_prepare.jsonl where upstream collate is compatible.
  render  [name ...]  Materialize prompt-ready rows for ARC-AGI/ARC-AGI-2/GSM8K.
  rsi     [name ...]  Recreate the ARC-AGI and GAIA RSI train/holdout splits.
  all                  Run prepare, collate, render and rsi for all supported benchmarks.

  `collate` supports aegis_v4_safety, arc-agi, arc-agi-v2 and gsm8k. GAIA rows already
  carry responses_create_params.input, and ARC-AGI-3 has a self-contained agent config;
  both use their prepare output directly instead of passing through this collate step.

Names:
  all, aegis_v4_safety, arc-agi, arc-agi-v2, arc-agi-v3, gaia, gsm8k

Environment:
  GYM_BIN                 Command to invoke (default: agent-gym).
  PYTHON_BIN              Python used for split/resolve helpers.
  AGENT_GYM_PREPARE_OUTPUT
                          Collate output directory (default: .tmp/data-prep).

Notes:
  * No benchmark or component data is tracked in Git. `prepare arc-agi` also
    regenerates the base ARC example fixture used by arc_agi_simple_agent.
  * `all` may download public datasets and GAIA requires prior Hugging Face access.
  * pytest and `gym env start` do not prepare benchmark data automatically.
EOF
}

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

run() {
    if ((dry_run)); then
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
    else
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
        "$@"
    fi
}

is_supported() {
    local wanted="$1"
    local candidate
    for candidate in "${all_benchmarks[@]}"; do
        [[ "${candidate}" == "${wanted}" ]] && return 0
    done
    return 1
}

is_renderable() {
    local wanted="$1"
    local candidate
    for candidate in "${renderable_benchmarks[@]}"; do
        [[ "${candidate}" == "${wanted}" ]] && return 0
    done
    return 1
}

is_collateable() {
    local wanted="$1"
    local candidate
    for candidate in "${collateable_benchmarks[@]}"; do
        [[ "${candidate}" == "${wanted}" ]] && return 0
    done
    return 1
}

is_rsi_benchmark() {
    local wanted="$1"
    local candidate
    for candidate in "${rsi_benchmarks[@]}"; do
        [[ "${candidate}" == "${wanted}" ]] && return 0
    done
    return 1
}

expand_targets() {
    local defaults="$1"
    shift
    local target
    if (($# == 0)); then
        printf '%s\n' ${defaults}
        return
    fi
    for target in "$@"; do
        if [[ "${target}" == "all" ]]; then
            printf '%s\n' ${defaults}
        else
            printf '%s\n' "${target}"
        fi
    done | awk '!seen[$0]++'
}

benchmark_jsonl() {
    case "$1" in
        aegis_v4_safety) printf '%s\n' 'benchmarks/aegis_v4_safety/data/aegis_v4_safety_benchmark.jsonl' ;;
        arc-agi) printf '%s\n' 'benchmarks/arc-agi/data/arc_agi_benchmark.jsonl' ;;
        arc-agi-v2) printf '%s\n' 'benchmarks/arc-agi-v2/data/arc_agi_v2_benchmark.jsonl' ;;
        arc-agi-v3) printf '%s\n' 'benchmarks/arc-agi-v3/data/arc_agi_v3_games.jsonl' ;;
        gaia) printf '%s\n' 'benchmarks/gaia/data/gaia_text_validation.jsonl' ;;
        gsm8k) printf '%s\n' 'benchmarks/gsm8k/data/gsm8k_benchmark.jsonl' ;;
        *) die "no JSONL mapping for benchmark ${1}" ;;
    esac
}

prompt_config() {
    case "$1" in
        arc-agi|arc-agi-v2) printf '%s\n' 'benchmarks/prompts/generic/arc.yaml' ;;
        gsm8k) printf '%s\n' 'benchmarks/prompts/generic/math.yaml' ;;
        *) die "no prompt mapping for benchmark ${1}" ;;
    esac
}

seed_arc_agi_example() {
    local relative='resources_servers/arc_agi/data/example.jsonl'
    local destination="${repo_root}/${relative}"
    [[ -f "${destination}" ]] && return
    if ((dry_run)); then
        log "+ generate ${relative} with resources_servers/arc_agi/prepare_example.py"
        return
    fi
    run "${PYTHON_BIN}" scripts/run_component.py resources_servers/arc_agi/prepare_example.py
}

seed_prepare_input() {
    local benchmark="$1"
    local relative
    case "${benchmark}" in
        arc-agi)
            seed_arc_agi_example
            return
            ;;
        aegis_v4_safety)
            relative='benchmarks/aegis_v4_safety/data/source.jsonl'
            ;;
        *)
            return
            ;;
    esac

    local destination="${repo_root}/${relative}"
    [[ -f "${destination}" ]] && return
    if ((dry_run)); then
        log "+ seed ${relative} from the pinned wheel when missing"
        return
    fi

    local source
    source="$("${PYTHON_BIN}" - "${relative}" <<'PY'
import sys
from pathlib import Path

from agent_gym.layout import GymLayout

layout = GymLayout.discover()
path = layout.resolve_data(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"resolved data asset is not a file: {path}")
print(path)
PY
    )" || die "cannot resolve ${relative} from overlay/repo/wheel"

    mkdir -p "$(dirname "${destination}")"
    cp "${source}" "${destination}"
    log "Seeded ${relative} from ${source}"
}

prepare_benchmarks() {
    local targets=("$@")
    local benchmark
    for benchmark in "${targets[@]}"; do
        is_supported "${benchmark}" || die "unknown benchmark: ${benchmark}"
        seed_prepare_input "${benchmark}"
        # Prepare one benchmark per process. A combined invocation resolves every benchmark
        # config at once; v1/v2 share the same resources-server path and upstream deduplicates
        # them, silently preparing only one ARC variant.
        run "${gym_bin}" eval prepare --benchmark "${benchmark}"
    done
}

collate_benchmarks() {
    local benchmark
    local data_path
    for benchmark in "$@"; do
        is_supported "${benchmark}" || die "unknown benchmark: ${benchmark}"
        is_collateable "${benchmark}" || die "collate is not supported for ${benchmark}; use prepare output directly (GAIA already carries model input; ARC-AGI-3 is self-contained)"
        data_path="${repo_root}/$(benchmark_jsonl "${benchmark}")"
        if [[ ! -f "${data_path}" && ${dry_run} -eq 0 ]]; then
            die "missing ${data_path#"${repo_root}/"}; run: bash scripts/prepare_data.sh prepare ${benchmark}"
        fi
        run "${gym_bin}" dataset collate \
            --config "benchmarks/${benchmark}/config.yaml" \
            --output-dir "${output_root}/${benchmark}" \
            --mode train_preparation
    done
}

render_benchmarks() {
    local benchmark
    local data_path
    local output_path
    local prompt_path
    for benchmark in "$@"; do
        is_renderable "${benchmark}" || die "render is not defined for ${benchmark}; supported: ${renderable_benchmarks[*]}"
        data_path="${repo_root}/$(benchmark_jsonl "${benchmark}")"
        if [[ ! -f "${data_path}" && ${dry_run} -eq 0 ]]; then
            die "missing ${data_path#"${repo_root}/"}; run: bash scripts/prepare_data.sh prepare ${benchmark}"
        fi
        prompt_path="${AGENT_GYM_OVERLAY}/$(prompt_config "${benchmark}")"
        output_path="${data_path%.jsonl}_materialized.jsonl"
        run "${gym_bin}" dataset render \
            --input "${data_path}" \
            --prompt-config "${prompt_path}" \
            --output "${output_path}"
    done
}

rsi_splits() {
    local benchmark
    local source_path
    local output_dir
    local splitter
    splitter='benchmarks/rsi_dataset_tools/split_jsonl.py'
    for benchmark in "$@"; do
        is_rsi_benchmark "${benchmark}" || die "RSI split is not defined for ${benchmark}; supported: ${rsi_benchmarks[*]}"
        case "${benchmark}" in
            arc-agi)
                source_path="${repo_root}/benchmarks/arc-agi/data/arc_agi_benchmark_materialized.jsonl"
                output_dir="${repo_root}/benchmarks/arc-agi/data/rsi"
                ;;
            gaia)
                source_path="${repo_root}/benchmarks/gaia/data/gaia_text_validation.jsonl"
                output_dir="${repo_root}/benchmarks/gaia/data/rsi"
                ;;
        esac
        if [[ ! -f "${source_path}" && ${dry_run} -eq 0 ]]; then
            die "missing ${source_path#"${repo_root}/"}; run prepare and render first"
        fi
        run "${PYTHON_BIN}" scripts/run_component.py "${splitter}" \
            "${source_path}" \
            --train "${output_dir}/${benchmark//-/_}_train.jsonl" \
            --holdout "${output_dir}/${benchmark//-/_}_holdout.jsonl" \
            --id-field task_id
    done
}

if (($# == 0)); then
    usage
    exit 2
fi

args=()
for argument in "$@"; do
    case "${argument}" in
        -h|--help)
            usage
            exit 0
            ;;
        --dry-run)
            dry_run=1
            ;;
        *)
            args+=("${argument}")
            ;;
    esac
done
set -- "${args[@]}"
command="${1:-}"
shift || true

case "${command}" in
    prepare)
        mapfile -t targets < <(expand_targets "${all_benchmarks[*]}" "$@")
        prepare_benchmarks "${targets[@]}"
        ;;
    collate)
        mapfile -t targets < <(expand_targets "${collateable_benchmarks[*]}" "$@")
        collate_benchmarks "${targets[@]}"
        ;;
    render)
        mapfile -t targets < <(expand_targets "${renderable_benchmarks[*]}" "$@")
        render_benchmarks "${targets[@]}"
        ;;
    rsi)
        mapfile -t targets < <(expand_targets "${rsi_benchmarks[*]}" "$@")
        rsi_splits "${targets[@]}"
        ;;
    all)
        (($# == 0)) || die "all does not accept benchmark names; run individual commands for a subset"
        prepare_benchmarks "${all_benchmarks[@]}"
        collate_benchmarks "${collateable_benchmarks[@]}"
        render_benchmarks "${renderable_benchmarks[@]}"
        rsi_splits "${rsi_benchmarks[@]}"
        ;;
    *)
        usage >&2
        die "unknown command: ${command}"
        ;;
esac
