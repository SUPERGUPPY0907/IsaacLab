#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${ISAACLAB_ROOT_DEFAULT}}"
RSL_RL_ROOT="${RSL_RL_ROOT:-/home/superguppy/rsl_rl}"

DEFAULT_TASKS="Isaac-Ant-v0,Isaac-Humanoid-v0,Isaac-Lift-Cube-Franka-v0,Isaac-Open-Drawer-Franka-v0,Isaac-Velocity-Rough-Anymal-D-v0,Isaac-Velocity-Rough-Unitree-Go2-v0,Isaac-Velocity-Rough-G1-v0,Isaac-Tracking-LocoManip-Digit-v0"
TASKS="${TASKS:-${TASK:-${DEFAULT_TASKS}}}"
AGENT="${AGENT:-belm_genpo}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
HEADLESS="${HEADLESS:-1}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-belm_genpo_10seed}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${ISAACLAB_ROOT}/logs/belm_genpo_multi_task_10seed_sweeps}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
SLOT_POLL_INTERVAL="${SLOT_POLL_INTERVAL:-10}"

LAG_COEFF="${LAG_COEFF:-}"
A_COEFF="${A_COEFF:-}"
B_COEFF="${B_COEFF:-}"
EPS_COEFF="${EPS_COEFF:-}"

TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

usage() {
    cat <<EOF
Usage:
  bash ${0##*/} [--skip-install]

Environment overrides:
  ISAACLAB_ROOT        IsaacLab root directory. Default: ${ISAACLAB_ROOT_DEFAULT}
  RSL_RL_ROOT          Local rsl_rl checkout. Default: /home/superguppy/rsl_rl
  TASKS                Comma-separated tasks. Default: ${DEFAULT_TASKS}
  TASK                 Backward-compatible single-task override when TASKS is unset.
  AGENT                BELM agent key. Default: belm_genpo
                       Supported: belm_genpo, belmgenpo
  SEEDS                Comma-separated seeds. Default: 42,43,44,45,46,47,48,49,50,51
  GPU_IDS              Comma-separated GPU ids. Default: 0,1,2,3,4,5,6,7
  HEADLESS             1 to add --headless, 0 otherwise. Default: 1
  NUM_ENVS             Optional --num_envs override.
  MAX_ITERATIONS       Optional --max_iterations override.
  SAVE_INTERVAL        Optional Hydra override for agent.save_interval.
  EXPERIMENT_PREFIX    Experiment name prefix. Default: belm_genpo_10seed
  WANDB_PROJECT_PREFIX Optional prefix for generated wandb projects.
                       Project format: <prefix_>task_slug_belm_genpo
  LAUNCH_LOG_DIR       Per-run log directory. Default: logs/belm_genpo_multi_task_10seed_sweeps
  INSTALL_EDITABLE     1 to install local rsl_rl into IsaacLab env before running.
  SLOT_POLL_INTERVAL   Seconds to wait before checking for a free GPU slot. Default: 10

BELM policy overrides:
  LAG_COEFF            Tied BELM coefficient. When set, force:
                       agent.policy.a_coeff=null
                       agent.policy.b_coeff=null
                       agent.policy.eps_coeff=null
                       agent.policy.lag_coeff=<value>
  A_COEFF              Untied BELM a coefficient. If any of A/B/EPS is set, force:
  B_COEFF                agent.policy.lag_coeff=null
  EPS_COEFF            Unspecified untied coeffs keep task/default config values.
                       Do not combine LAG_COEFF with A_COEFF/B_COEFF/EPS_COEFF.

Examples:
  bash ${0##*/}
  TASKS=Isaac-Humanoid-v0 A_COEFF=0.25 B_COEFF=0.75 EPS_COEFF=1.0 bash ${0##*/}
  TASKS=Isaac-Humanoid-v0 LAG_COEFF=0.97 SEEDS=0,1 GPU_IDS=0,1 bash ${0##*/}
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
        Isaac-Ant-v0) echo "ant" ;;
        Isaac-Humanoid-v0) echo "humanoid" ;;
        Isaac-Lift-Cube-Franka-v0) echo "franka_lift_cube" ;;
        Isaac-Open-Drawer-Franka-v0) echo "franka_open_drawer" ;;
        Isaac-Velocity-Rough-Anymal-D-v0) echo "anymal_d_rough" ;;
        Isaac-Velocity-Rough-Unitree-Go2-v0) echo "unitree_go2_rough" ;;
        Isaac-Velocity-Rough-H1-v0) echo "h1_rough" ;;
        Isaac-Velocity-Rough-G1-v0) echo "g1_rough" ;;
        Isaac-Tracking-LocoManip-Digit-v0) echo "digit_loco_manip" ;;
        *)
            echo "Unsupported task for BELM-GenPO sweep: ${task}" >&2
            exit 1
            ;;
    esac
}

canonical_agent() {
    local agent="$1"
    case "${agent}" in
        belmgenpo | belm_genpo) echo "belm_genpo" ;;
        *)
            echo "Unsupported agent '${agent}'. Use belm_genpo or belmgenpo." >&2
            exit 1
            ;;
    esac
}

has_untied_policy_override() {
    [[ -n "$(trim_value "${A_COEFF}")" || -n "$(trim_value "${B_COEFF}")" || -n "$(trim_value "${EPS_COEFF}")" ]]
}

validate_policy_overrides() {
    local lag_value
    lag_value="$(trim_value "${LAG_COEFF}")"

    if [[ -n "${lag_value}" ]] && has_untied_policy_override; then
        echo "Do not combine LAG_COEFF with A_COEFF/B_COEFF/EPS_COEFF." >&2
        exit 1
    fi
}

policy_mode() {
    local lag_value
    lag_value="$(trim_value "${LAG_COEFF}")"

    if [[ -n "${lag_value}" ]]; then
        echo "tied"
    elif has_untied_policy_override; then
        echo "untied"
    else
        echo "default"
    fi
}

policy_variant_label() {
    local mode
    mode="$(policy_mode)"

    case "${mode}" in
        default)
            echo "default"
            ;;
        tied)
            echo "lag$(sanitize_value "${LAG_COEFF}")"
            ;;
        untied)
            local -a parts=()

            if [[ -n "$(trim_value "${A_COEFF}")" ]]; then
                parts+=("a$(sanitize_value "${A_COEFF}")")
            fi
            if [[ -n "$(trim_value "${B_COEFF}")" ]]; then
                parts+=("b$(sanitize_value "${B_COEFF}")")
            fi
            if [[ -n "$(trim_value "${EPS_COEFF}")" ]]; then
                parts+=("eps$(sanitize_value "${EPS_COEFF}")")
            fi

            local joined
            joined="$(IFS=_; echo "${parts[*]}")"
            echo "${joined}"
            ;;
    esac
}

build_wandb_project() {
    local task_label="$1"

    if [[ -n "${WANDB_PROJECT_PREFIX}" ]]; then
        echo "${WANDB_PROJECT_PREFIX}_${task_label}_belm_genpo"
    else
        echo "${task_label}_belm_genpo"
    fi
}

build_experiment_name() {
    local task_label="$1"
    local seed="$2"
    local prefix=""
    local variant_label

    variant_label="belm_genpo"
    if [[ "$(policy_mode)" != "default" ]]; then
        variant_label="${variant_label}_$(policy_variant_label)"
    fi

    if [[ -n "${EXPERIMENT_PREFIX}" ]]; then
        prefix="${EXPERIMENT_PREFIX}_"
    fi

    echo "${prefix}${task_label}_${variant_label}_seed$(sanitize_value "${seed}")"
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
    local agent_entry="$3"
    local experiment_name="$4"
    local wandb_project="$5"
    local label="$6"
    shift 6

    local gpu_id="${gpu_ids[$slot_index]}"
    local env_device="cuda:${gpu_id}"
    local log_file="${LAUNCH_LOG_DIR}/${experiment_name}.log"
    local status_file="${LAUNCH_LOG_DIR}/${experiment_name}.status"

    local -a run_args=(
        --task "${task}"
        --agent "${agent_entry}"
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
    echo "Agent entry: ${agent_entry}"
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

append_policy_overrides() {
    local -n out_ref="$1"
    local lag_value
    local a_value
    local b_value
    local eps_value

    lag_value="$(trim_value "${LAG_COEFF}")"
    a_value="$(trim_value "${A_COEFF}")"
    b_value="$(trim_value "${B_COEFF}")"
    eps_value="$(trim_value "${EPS_COEFF}")"

    if [[ -n "${lag_value}" ]]; then
        out_ref+=(
            "agent.policy.a_coeff=null"
            "agent.policy.b_coeff=null"
            "agent.policy.eps_coeff=null"
            "agent.policy.lag_coeff=${lag_value}"
        )
        return 0
    fi

    if [[ -n "${a_value}" || -n "${b_value}" || -n "${eps_value}" ]]; then
        out_ref+=("agent.policy.lag_coeff=null")
        [[ -n "${a_value}" ]] && out_ref+=("agent.policy.a_coeff=${a_value}")
        [[ -n "${b_value}" ]] && out_ref+=("agent.policy.b_coeff=${b_value}")
        [[ -n "${eps_value}" ]] && out_ref+=("agent.policy.eps_coeff=${eps_value}")
    fi
}

schedule_variant() {
    local task="$1"
    local task_label="$2"
    local seed="$3"

    local agent_entry
    local experiment_name
    local wandb_project
    local label
    local -a overrides=()

    agent_entry="$(canonical_agent "${AGENT}")"
    experiment_name="$(build_experiment_name "${task_label}" "${seed}")"
    wandb_project="$(build_wandb_project "${task_label}")"
    label="belm_genpo/$(policy_mode)"

    overrides+=("agent.seed=${seed}")
    append_policy_overrides overrides

    schedule_run "${task}" "${agent_entry}" "${experiment_name}" "${wandb_project}" "${label}" "${overrides[@]}"
}

trap 'terminate_running_jobs; exit 130' INT TERM

declare -a tasks=()
declare -a seeds=()

validate_policy_overrides
csv_to_array "${TASKS}" tasks
csv_to_array "${SEEDS}" seeds
initialize_gpu_ids

if [[ ${#tasks[@]} -eq 0 || ${#seeds[@]} -eq 0 ]]; then
    echo "TASKS and SEEDS must both be non-empty." >&2
    exit 1
fi

mkdir -p "${LAUNCH_LOG_DIR}"

echo "IsaacLab root: ${ISAACLAB_ROOT}"
echo "rsl_rl root: ${RSL_RL_ROOT}"
echo "Tasks: ${TASKS}"
echo "Agent: $(canonical_agent "${AGENT}")"
echo "Seeds: ${SEEDS}"
echo "GPU ids: ${gpu_ids[*]}"
echo "Max iterations override: ${MAX_ITERATIONS:-<task cfg default>}"
echo "Save interval override: ${SAVE_INTERVAL:-<task cfg default>}"
echo "Num envs override: ${NUM_ENVS:-<task cfg default>}"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "wandb project prefix: ${WANDB_PROJECT_PREFIX:-<none>}"
echo "Launcher log dir: ${LAUNCH_LOG_DIR}"
echo "BELM policy mode: $(policy_mode)"
echo "LAG_COEFF: ${LAG_COEFF:-<unset>}"
echo "A_COEFF: ${A_COEFF:-<unset>}"
echo "B_COEFF: ${B_COEFF:-<unset>}"
echo "EPS_COEFF: ${EPS_COEFF:-<unset>}"

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

    echo
    echo "==== Starting BELM-GenPO 10-seed sweep for ${task} (${task_label}) ===="

    for seed in "${seeds[@]}"; do
        seed="$(trim_value "${seed}")"
        [[ -z "${seed}" ]] && continue
        schedule_variant "${task}" "${task_label}" "${seed}"
    done
done

wait_for_all_jobs

echo
if [[ ${failed_jobs} -ne 0 ]]; then
    echo "BELM-GenPO multi-task 10-seed sweep finished with ${failed_jobs} failed job(s)." >&2
    exit 1
fi
echo "BELM-GenPO multi-task 10-seed sweep finished successfully."
