#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${ISAACLAB_ROOT_DEFAULT}}"
RSL_RL_ROOT="${RSL_RL_ROOT:-/home/superguppy/rsl_rl}"

TASK="${TASK:-Isaac-Velocity-Rough-G1-v0}"
AGENT="${AGENT:-belm_genpo}"
SEEDS="${SEEDS:-42}"
FLOW_STEPS_SWEEP="${FLOW_STEPS_SWEEP:-4,8,16,32,64}"
NUM_LEARNING_EPOCHS_SWEEP="${NUM_LEARNING_EPOCHS_SWEEP:-5,16,32,48}"
SCHEDULE_SWEEP="${SCHEDULE_SWEEP:-adaptive,fixed}"
FIXED_LR_SWEEP="${FIXED_LR_SWEEP:-1e-4,5e-4,1e-3}"

HEADLESS="${HEADLESS:-1}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000000000000000}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-g1_flow_sweep}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${ISAACLAB_ROOT}/logs/g1_flow_sweeps}"
GPU_IDS="${GPU_IDS:-}"
SLOT_POLL_INTERVAL="${SLOT_POLL_INTERVAL:-10}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

usage() {
    cat <<EOF
Usage:
  bash ${0##*/} [--skip-install]

Environment overrides:
  ISAACLAB_ROOT             IsaacLab root directory. Default: ${ISAACLAB_ROOT_DEFAULT}
  RSL_RL_ROOT               Local rsl_rl checkout. Used only when INSTALL_EDITABLE=1.
  TASK                      IsaacLab task. Default: Isaac-Velocity-Rough-G1-v0
  AGENT                     Flow-based IsaacLab agent key. Default: belm_genpo
                            Supported: genpo, belmgenpo, belm_genpo, genpo_pp, genpo++, sgenpo
  SEEDS                     Comma-separated seeds. Default: 42
  FLOW_STEPS_SWEEP          Comma-separated policy.flow_num_steps values. Default: 5,10,15
  NUM_LEARNING_EPOCHS_SWEEP Comma-separated algorithm.num_learning_epochs values. Default: 16,32,48
  SCHEDULE_SWEEP            Comma-separated schedule modes. Default: adaptive,fixed
  FIXED_LR_SWEEP            Comma-separated learning rates used only when schedule=fixed.
                            Default: 1e-4,5e-4,1e-3
  HEADLESS                  1 to add --headless, 0 otherwise. Default: 1
  NUM_ENVS                  Optional --num_envs override.
  MAX_ITERATIONS            Optional --max_iterations override.
  SAVE_INTERVAL             Optional Hydra override for agent.save_interval.
  EXPERIMENT_PREFIX         Prefix added to generated experiment_name values. Default: g1_flow_sweep
  LAUNCH_LOG_DIR            Directory for per-run launcher logs. Default: logs/g1_flow_sweeps
  GPU_IDS                   Comma-separated GPU ids. Default: auto-detect visible GPUs.
  SLOT_POLL_INTERVAL        Seconds between free-slot checks. Default: 10
  INSTALL_EDITABLE          1 to install local rsl_rl into IsaacLab env before running. Default: 0
  WANDB_PROJECT             Optional override. Default:
                            <TASK>_flow_num_steps_num_learning_epochs_lr_schedule_sweep

Examples:
  bash ${0##*/}
  FLOW_STEPS_SWEEP=5,10 NUM_LEARNING_EPOCHS_SWEEP=16,32 FIXED_LR_SWEEP=5e-4,1e-3 bash ${0##*/}
  AGENT=genpo SEEDS=1,2 GPU_IDS=0,1 bash ${0##*/}
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
    task="${task//-/_}"
    task="${task// /_}"
    echo "${task,,}"
}

agent_slug() {
    local agent="$1"
    agent="${agent//+/_plus_}"
    agent="${agent//-/_}"
    echo "${agent,,}"
}

validate_flow_agent() {
    case "${AGENT}" in
        genpo | belmgenpo | belm_genpo | genpo_pp | genpo++ | sgenpo)
            ;;
        *)
            echo "AGENT='${AGENT}' does not expose agent.policy.flow_num_steps for this sweep." >&2
            echo "Use one of: genpo, belmgenpo, belm_genpo, genpo_pp, genpo++, sgenpo" >&2
            exit 1
            ;;
    esac
}

build_wandb_project() {
    local task="$1"
    if [[ -n "${WANDB_PROJECT}" ]]; then
        echo "${WANDB_PROJECT}"
    else
        echo "${task}_flow_num_steps_num_learning_epochs_lr_schedule_sweep"
    fi
}

build_variant_name() {
    local flow_steps="$1"
    local epochs="$2"
    local schedule="$3"
    local learning_rate="${4:-}"

    local variant="flow$(sanitize_value "${flow_steps}")_epochs$(sanitize_value "${epochs}")"
    if [[ "${schedule}" == "adaptive" ]]; then
        variant="${variant}_schedadaptive_lrcfg"
    else
        variant="${variant}_schedfixed_lr$(sanitize_value "${learning_rate}")"
    fi
    echo "${variant}"
}

build_experiment_name() {
    local task="$1"
    local variant="$2"
    local seed="$3"
    local prefix=""

    if [[ -n "${EXPERIMENT_PREFIX}" ]]; then
        prefix="${EXPERIMENT_PREFIX}_"
    fi

    echo "${prefix}$(task_slug "${task}")_$(agent_slug "${AGENT}")_${variant}_seed${seed}"
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
    local wandb_project="$2"
    local seed="$3"
    local flow_steps="$4"
    local epochs="$5"
    local schedule="$6"
    local learning_rate="${7:-}"

    local variant
    variant="$(build_variant_name "${flow_steps}" "${epochs}" "${schedule}" "${learning_rate}")"

    local -a overrides=(
        "agent.seed=${seed}"
        "agent.policy.flow_num_steps=${flow_steps}"
        "agent.algorithm.num_learning_epochs=${epochs}"
        "agent.algorithm.schedule=${schedule}"
    )

    if [[ "${schedule}" == "fixed" ]]; then
        overrides+=("agent.algorithm.learning_rate=${learning_rate}")
    fi

    schedule_run "${task}" "$(build_experiment_name "${task}" "${variant}" "${seed}")" "${wandb_project}" "${variant}" \
        "${overrides[@]}"
}

trap 'terminate_running_jobs; exit 130' INT TERM

declare -a seeds=()
declare -a flow_steps_values=()
declare -a epoch_values=()
declare -a schedule_modes=()
declare -a fixed_lr_values=()

validate_flow_agent
csv_to_array "${SEEDS}" seeds
csv_to_array "${FLOW_STEPS_SWEEP}" flow_steps_values
csv_to_array "${NUM_LEARNING_EPOCHS_SWEEP}" epoch_values
csv_to_array "${SCHEDULE_SWEEP}" schedule_modes
csv_to_array "${FIXED_LR_SWEEP}" fixed_lr_values
initialize_gpu_ids

if [[ ${#seeds[@]} -eq 0 || ${#flow_steps_values[@]} -eq 0 || ${#epoch_values[@]} -eq 0 || ${#schedule_modes[@]} -eq 0 ]]; then
    echo "SEEDS, FLOW_STEPS_SWEEP, NUM_LEARNING_EPOCHS_SWEEP, and SCHEDULE_SWEEP must all be non-empty." >&2
    exit 1
fi

mkdir -p "${LAUNCH_LOG_DIR}"

wandb_project="$(build_wandb_project "${TASK}")"

echo "IsaacLab root: ${ISAACLAB_ROOT}"
echo "rsl_rl root: ${RSL_RL_ROOT}"
echo "Task: ${TASK}"
echo "Agent: ${AGENT}"
echo "Seeds: ${SEEDS}"
echo "Flow steps sweep: ${FLOW_STEPS_SWEEP}"
echo "Num learning epochs sweep: ${NUM_LEARNING_EPOCHS_SWEEP}"
echo "Schedule sweep: ${SCHEDULE_SWEEP}"
echo "Fixed LR sweep: ${FIXED_LR_SWEEP}"
echo "GPU ids: ${gpu_ids[*]}"
echo "Max iterations override: ${MAX_ITERATIONS:-<task cfg default>}"
echo "Save interval override: ${SAVE_INTERVAL:-<task cfg default>}"
echo "Num envs override: ${NUM_ENVS:-<task cfg default>}"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "Launcher log dir: ${LAUNCH_LOG_DIR}"
echo "wandb_project: ${wandb_project}"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
    echo
    echo "==== Installing local rsl_rl into IsaacLab Python environment ===="
    (
        cd "${ISAACLAB_ROOT}"
        ./isaaclab.sh -p -m pip install -e "${RSL_RL_ROOT}"
    )
fi

for seed in "${seeds[@]}"; do
    seed="$(trim_value "${seed}")"
    [[ -z "${seed}" ]] && continue

    for flow_steps in "${flow_steps_values[@]}"; do
        flow_steps="$(trim_value "${flow_steps}")"
        [[ -z "${flow_steps}" ]] && continue

        for epochs in "${epoch_values[@]}"; do
            epochs="$(trim_value "${epochs}")"
            [[ -z "${epochs}" ]] && continue

            for schedule in "${schedule_modes[@]}"; do
                schedule="$(trim_value "${schedule}")"
                [[ -z "${schedule}" ]] && continue

                case "${schedule}" in
                    adaptive)
                        schedule_variant "${TASK}" "${wandb_project}" "${seed}" "${flow_steps}" "${epochs}" "adaptive"
                        ;;
                    fixed)
                        if [[ ${#fixed_lr_values[@]} -eq 0 ]]; then
                            echo "FIXED_LR_SWEEP must be non-empty when SCHEDULE_SWEEP includes 'fixed'." >&2
                            exit 1
                        fi

                        for learning_rate in "${fixed_lr_values[@]}"; do
                            learning_rate="$(trim_value "${learning_rate}")"
                            [[ -z "${learning_rate}" ]] && continue
                            schedule_variant "${TASK}" "${wandb_project}" "${seed}" "${flow_steps}" "${epochs}" "fixed" "${learning_rate}"
                        done
                        ;;
                    *)
                        echo "Unsupported schedule mode '${schedule}'. Use adaptive or fixed." >&2
                        exit 1
                        ;;
                esac
            done
        done
    done
done

wait_for_all_jobs

echo
if [[ ${failed_jobs} -ne 0 ]]; then
    echo "G1 flow sweep finished with ${failed_jobs} failed job(s)." >&2
    exit 1
fi
echo "G1 flow sweep finished successfully."
