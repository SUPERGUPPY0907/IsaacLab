#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${ISAACLAB_ROOT_DEFAULT}}"
RSL_RL_ROOT="${RSL_RL_ROOT:-/home/superguppy/rsl_rl}"

DEFAULT_TASKS="Isaac-Humanoid-v0"
TASKS="${TASKS:-${TASK:-${DEFAULT_TASKS}}}"
AGENT="${AGENT:-belmgenpo}"
SEEDS="${SEEDS:-42}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
HEADLESS="${HEADLESS:-1}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-belmgenpo_alg_sweep}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${ISAACLAB_ROOT}/logs/belmgenpo_ema_trust_region_latent_noise_sweeps}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
SLOT_POLL_INTERVAL="${SLOT_POLL_INTERVAL:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

EMA_DECAY_SWEEP="${EMA_DECAY_SWEEP:-${EMA_SWEEP:-0.0,0.95,0.99}}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-500}"
TRUST_REGION_MODE_SWEEP="${TRUST_REGION_MODE_SWEEP:-${TRUST_REGION_TYPE_SWEEP:-ppo,spo,aspo}}"
LATENT_NOISE_SWEEP="${LATENT_NOISE_SWEEP:-${STORAGE_LATENT_NOISE_SWEEP:-0.0,0.01,0.02}}"

TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

usage() {
    cat <<EOF
Usage:
  bash ${0##*/} [--skip-install]

Environment overrides:
  ISAACLAB_ROOT           IsaacLab root directory. Default: ${ISAACLAB_ROOT_DEFAULT}
  RSL_RL_ROOT             Local rsl_rl checkout. Default: /home/superguppy/rsl_rl
  TASKS                   Comma-separated tasks. Default: ${DEFAULT_TASKS}
  TASK                    Backward-compatible single-task override.
  AGENT                   BELMGenPO agent key. Default: belmgenpo
                          Supported: belmgenpo, belm_genpo
  SEEDS                   Comma-separated seeds. Default: 42
  GPU_IDS                 Comma-separated GPU ids. Default: 0,1,2,3,4,5,6,7
  HEADLESS                1 to add --headless, 0 otherwise. Default: 1
  NUM_ENVS                Optional --num_envs override.
  MAX_ITERATIONS          Optional --max_iterations override.
  SAVE_INTERVAL           Optional Hydra override for agent.save_interval.
  EXPERIMENT_PREFIX       Experiment name prefix. Default: belmgenpo_alg_sweep
  LAUNCH_LOG_DIR          Per-run log directory. Default: logs/belmgenpo_ema_trust_region_latent_noise_sweeps
  INSTALL_EDITABLE        1 to install local rsl_rl into IsaacLab env before running.
  SLOT_POLL_INTERVAL      Seconds to wait before checking for a free GPU slot. Default: 10
  WANDB_PROJECT           Optional override. Default: <task>_belmgenpo_ema_trust_region_latent_noise_sweep

Algorithm sweep overrides:
  EMA_DECAY_SWEEP         Comma-separated agent.algorithm.ema_decay values.
                          Alias: EMA_SWEEP
                          Default: 0.0,0.95,0.99
  EMA_WARMUP_STEPS        Fixed agent.algorithm.ema_warmup_steps override for all runs.
                          Default: 500
  TRUST_REGION_MODE_SWEEP Comma-separated agent.algorithm.trust_region_mode values.
                          Alias: TRUST_REGION_TYPE_SWEEP
                          Default: ppo,spo,aspo
  LATENT_NOISE_SWEEP      Comma-separated agent.algorithm.storage_latent_noise_std values.
                          Alias: STORAGE_LATENT_NOISE_SWEEP
                          Default: 0.0,0.01,0.05

Examples:
  bash ${0##*/}
  SEEDS=1,2 GPU_IDS=0,1 EMA_DECAY_SWEEP=0.0,0.95,0.99 bash ${0##*/}
  TRUST_REGION_TYPE_SWEEP=ppo,aspo LATENT_NOISE_SWEEP=0.0,0.02,0.05 bash ${0##*/}
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--skip-install" ]]; then
    INSTALL_EDITABLE=0
fi

csv_to_array() {
    local csv="$1"
    local -n out_ref="$2"
    out_ref=()
    [[ -z "${csv}" ]] && return 0
    IFS=',' read -r -a out_ref <<< "${csv}"
}

trim_value() {
    local value="$1"
    value="${value// /}"
    echo "${value}"
}

sanitize_value() {
    local value
    value="$(trim_value "$1")"
    value="${value//-/neg}"
    value="${value//./p}"
    value="${value//+/plus}"
    echo "${value}"
}

task_slug() {
    local task="$1"
    case "${task}" in
        Isaac-Velocity-Rough-H1-v0) echo "h1_rough" ;;
        Isaac-Velocity-Rough-G1-v0) echo "g1_rough" ;;
        Isaac-Tracking-LocoManip-Digit-v0) echo "digit_loco_manip" ;;
        Isaac-Open-Drawer-Franka-v0) echo "franka_open_drawer" ;;
        Isaac-Humanoid-v0) echo "humanoid" ;;
        *)
            echo "Unsupported task for BELMGenPO algorithm sweep: ${task}" >&2
            exit 1
            ;;
    esac
}

validate_agent() {
    case "${AGENT}" in
        belmgenpo | belm_genpo)
            ;;
        *)
            echo "AGENT='${AGENT}' is not a supported BELMGenPO entry point for this sweep." >&2
            echo "Use one of: belmgenpo, belm_genpo" >&2
            exit 1
            ;;
    esac
}

build_wandb_project() {
    local task="$1"
    if [[ -n "${WANDB_PROJECT}" ]]; then
        echo "${WANDB_PROJECT}"
    else
        echo "$(task_slug "${task}")_belmgenpo_ema_trust_region_latent_noise_sweep"
    fi
}

build_variant_name() {
    local trust_region_mode="$1"
    local ema_decay="$2"
    local latent_noise="$3"

    echo "trust$(sanitize_value "${trust_region_mode}")_ema$(sanitize_value "${ema_decay}")_warm$(sanitize_value "${EMA_WARMUP_STEPS}")_latent$(sanitize_value "${latent_noise}")"
}

build_experiment_name() {
    local task_label="$1"
    local variant="$2"
    local seed="$3"
    local prefix=""

    if [[ -n "${EXPERIMENT_PREFIX}" ]]; then
        prefix="${EXPERIMENT_PREFIX}_"
    fi

    echo "${prefix}${task_label}_${variant}_seed${seed}"
}

declare -a gpu_ids=()
declare -a slot_pids=()
declare -a slot_experiments=()
declare -a slot_status_files=()
declare -i failed_jobs=0
next_slot_index=""

initialize_gpu_ids() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required to determine available GPUs." >&2
        exit 1
    fi

    if [[ -n "${GPU_IDS}" ]]; then
        csv_to_array "${GPU_IDS}" gpu_ids
    else
        mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    fi

    if [[ ${#gpu_ids[@]} -eq 0 ]]; then
        echo "No GPUs configured for the sweep." >&2
        exit 1
    fi

    local i
    for i in "${!gpu_ids[@]}"; do
        gpu_ids[$i]="$(trim_value "${gpu_ids[$i]}")"
        slot_pids[$i]=""
        slot_experiments[$i]=""
        slot_status_files[$i]=""
    done
}

terminate_running_jobs() {
    local pid
    for pid in "${slot_pids[@]}"; do
        [[ -z "${pid}" ]] && continue
        kill "${pid}" 2>/dev/null || true
    done
}

reap_finished_jobs() {
    local i pid status_file status gpu_id experiment_name

    for i in "${!gpu_ids[@]}"; do
        pid="${slot_pids[$i]}"
        status_file="${slot_status_files[$i]}"

        [[ -z "${pid}" || -z "${status_file}" || ! -f "${status_file}" ]] && continue

        status="$(<"${status_file}")"
        wait "${pid}" 2>/dev/null || true

        gpu_id="${gpu_ids[$i]}"
        experiment_name="${slot_experiments[$i]}"
        echo "[INFO] Experiment '${experiment_name}' on GPU ${gpu_id} finished with status ${status}"

        if [[ "${status}" -ne 0 ]]; then
            failed_jobs+=1
        fi

        rm -f "${status_file}"
        slot_pids[$i]=""
        slot_experiments[$i]=""
        slot_status_files[$i]=""
    done
}

wait_for_free_slot() {
    local i
    while true; do
        reap_finished_jobs
        for i in "${!gpu_ids[@]}"; do
            if [[ -z "${slot_pids[$i]}" ]]; then
                next_slot_index="${i}"
                return 0
            fi
        done
        sleep "${SLOT_POLL_INTERVAL}"
    done
}

launch_run() {
    local slot_index="$1"
    local task="$2"
    local experiment_name="$3"
    local wandb_project="$4"
    local label="$5"
    shift 5

    local gpu_id="${gpu_ids[$slot_index]}"
    local env_device="cuda:${gpu_id}"
    local log_file="${LAUNCH_LOG_DIR}/${experiment_name}.log"
    local status_file="${LAUNCH_LOG_DIR}/${experiment_name}.status"

    local -a run_args=(
        --task "${task}"
        --agent "${AGENT}"
        --device "${env_device}"
    )

    if [[ -n "${MAX_ITERATIONS}" ]]; then
        run_args+=(--max_iterations "${MAX_ITERATIONS}")
    fi
    if [[ "${HEADLESS}" == "1" ]]; then
        run_args+=(--headless)
    fi
    if [[ -n "${NUM_ENVS}" ]]; then
        run_args+=(--num_envs "${NUM_ENVS}")
    fi

    local -a hydra_args=(
        "agent.experiment_name=${experiment_name}"
        "agent.logger=wandb"
        "agent.wandb_project=${wandb_project}"
        "agent.device=${env_device}"
        "$@"
    )

    if [[ -n "${SAVE_INTERVAL}" ]]; then
        hydra_args+=("agent.save_interval=${SAVE_INTERVAL}")
    fi

    rm -f "${status_file}"

    echo
    echo "==== Launching ${task} :: ${label} on GPU ${gpu_id} ===="
    echo "Experiment name: ${experiment_name}"
    echo "wandb_project: ${wandb_project}"
    printf 'Overrides:'
    printf ' %q' "${hydra_args[@]}"
    printf '\n'

    (
        set +e
        cd "${ISAACLAB_ROOT}" || exit 1
        ./isaaclab.sh -p "${TRAIN_SCRIPT}" "${run_args[@]}" "${hydra_args[@]}"
        status=$?
        printf '%s\n' "${status}" > "${status_file}"
        exit "${status}"
    ) > "${log_file}" 2>&1 &

    slot_pids[$slot_index]="$!"
    slot_experiments[$slot_index]="${experiment_name}"
    slot_status_files[$slot_index]="${status_file}"

    echo "[INFO] Started '${experiment_name}' on GPU ${gpu_id} with pid ${slot_pids[$slot_index]}"
    echo "[INFO] Launcher log: ${log_file}"
}

schedule_run() {
    wait_for_free_slot
    launch_run "${next_slot_index}" "$@"
}

wait_for_all_jobs() {
    local slot_has_jobs
    while true; do
        reap_finished_jobs
        slot_has_jobs=0
        for pid in "${slot_pids[@]}"; do
            if [[ -n "${pid}" ]]; then
                slot_has_jobs=1
                break
            fi
        done
        [[ "${slot_has_jobs}" -eq 0 ]] && return 0
        sleep "${SLOT_POLL_INTERVAL}"
    done
}

schedule_variant() {
    local task="$1"
    local task_label="$2"
    local wandb_project="$3"
    local seed="$4"
    local trust_region_mode="$5"
    local ema_decay="$6"
    local latent_noise="$7"

    local variant
    variant="$(build_variant_name "${trust_region_mode}" "${ema_decay}" "${latent_noise}")"

    schedule_run "${task}" "$(build_experiment_name "${task_label}" "${variant}" "${seed}")" "${wandb_project}" "${variant}" \
        "agent.seed=${seed}" \
        "agent.algorithm.trust_region_mode=${trust_region_mode}" \
        "agent.algorithm.ema_decay=${ema_decay}" \
        "agent.algorithm.ema_warmup_steps=${EMA_WARMUP_STEPS}" \
        "agent.algorithm.storage_latent_noise_std=${latent_noise}"
}

trap 'terminate_running_jobs; exit 130' INT TERM

declare -a tasks=()
declare -a seeds=()
declare -a trust_region_modes=()
declare -a ema_decays=()
declare -a latent_noise_values=()

validate_agent
csv_to_array "${TASKS}" tasks
csv_to_array "${SEEDS}" seeds
csv_to_array "${TRUST_REGION_MODE_SWEEP}" trust_region_modes
csv_to_array "${EMA_DECAY_SWEEP}" ema_decays
csv_to_array "${LATENT_NOISE_SWEEP}" latent_noise_values
initialize_gpu_ids

if [[ ${#tasks[@]} -eq 0 || ${#seeds[@]} -eq 0 || ${#trust_region_modes[@]} -eq 0 || ${#ema_decays[@]} -eq 0 || ${#latent_noise_values[@]} -eq 0 ]]; then
    echo "TASKS, SEEDS, TRUST_REGION_MODE_SWEEP, EMA_DECAY_SWEEP, and LATENT_NOISE_SWEEP must all be non-empty." >&2
    exit 1
fi

mkdir -p "${LAUNCH_LOG_DIR}"

echo "IsaacLab root: ${ISAACLAB_ROOT}"
echo "rsl_rl root: ${RSL_RL_ROOT}"
echo "Tasks: ${TASKS}"
echo "Agent: ${AGENT}"
echo "Seeds: ${SEEDS}"
echo "GPU ids: ${gpu_ids[*]}"
echo "Max iterations override: ${MAX_ITERATIONS:-<task cfg default>}"
echo "Save interval override: ${SAVE_INTERVAL:-<task cfg default>}"
echo "Num envs override: ${NUM_ENVS:-<task cfg default>}"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "Launcher log dir: ${LAUNCH_LOG_DIR}"
echo "EMA decay sweep: ${EMA_DECAY_SWEEP}"
echo "EMA warmup steps: ${EMA_WARMUP_STEPS}"
echo "Trust-region mode sweep: ${TRUST_REGION_MODE_SWEEP}"
echo "Latent noise sweep: ${LATENT_NOISE_SWEEP}"
if [[ -n "${WANDB_PROJECT}" ]]; then
    echo "wandb_project override: ${WANDB_PROJECT}"
fi

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
    echo
    echo "==== Installing local rsl_rl into IsaacLab Python environment ===="
    (
        cd "${ISAACLAB_ROOT}"
        ./isaaclab.sh -p -m pip install -e "${RSL_RL_ROOT}"
    )
fi

for task in "${tasks[@]}"; do
    task="$(trim_value "${task}")"
    [[ -z "${task}" ]] && continue

    task_label="$(task_slug "${task}")"
    wandb_project="$(build_wandb_project "${task}")"

    echo
    echo "==== Starting BELMGenPO algorithm sweep for ${task} (${task_label}) ===="
    echo "wandb_project: ${wandb_project}"

    for seed in "${seeds[@]}"; do
        seed="$(trim_value "${seed}")"
        [[ -z "${seed}" ]] && continue

        for trust_region_mode in "${trust_region_modes[@]}"; do
            trust_region_mode="$(trim_value "${trust_region_mode}")"
            [[ -z "${trust_region_mode}" ]] && continue

            case "${trust_region_mode}" in
                ppo | spo | aspo)
                    ;;
                *)
                    echo "Unsupported trust-region mode '${trust_region_mode}'. Use ppo, spo, or aspo." >&2
                    exit 1
                    ;;
            esac

            for ema_decay in "${ema_decays[@]}"; do
                ema_decay="$(trim_value "${ema_decay}")"
                [[ -z "${ema_decay}" ]] && continue

                for latent_noise in "${latent_noise_values[@]}"; do
                    latent_noise="$(trim_value "${latent_noise}")"
                    [[ -z "${latent_noise}" ]] && continue

                    if [[ "${latent_noise}" == -* ]]; then
                        echo "LATENT_NOISE_SWEEP values must be non-negative, got '${latent_noise}'." >&2
                        exit 1
                    fi

                    schedule_variant "${task}" "${task_label}" "${wandb_project}" "${seed}" \
                        "${trust_region_mode}" "${ema_decay}" "${latent_noise}"
                done
            done
        done
    done
done

wait_for_all_jobs

echo
if [[ ${failed_jobs} -ne 0 ]]; then
    echo "BELMGenPO EMA/trust-region/latent-noise sweep finished with ${failed_jobs} failed job(s)." >&2
    exit 1
fi
echo "BELMGenPO EMA/trust-region/latent-noise sweep finished successfully."
