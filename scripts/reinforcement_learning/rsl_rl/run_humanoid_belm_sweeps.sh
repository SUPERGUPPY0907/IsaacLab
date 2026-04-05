#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RSL_RL_ROOT="${RSL_RL_ROOT:-/home/superguppy/rsl_rl}"

DEFAULT_TASKS="Isaac-Velocity-Rough-H1-v0,Isaac-Velocity-Rough-G1-v0,Isaac-Tracking-LocoManip-Digit-v0,Isaac-Open-Drawer-Franka-v0,Isaac-Humanoid-v0"
TASKS="${TASKS:-${TASK:-${DEFAULT_TASKS}}}"
AGENT="${AGENT:-belm_genpo}"
SEEDS="${SEEDS:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-belm_sweep}"
HEADLESS="${HEADLESS:-1}"
NUM_ENVS="${NUM_ENVS:-}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
A_SWEEP="${A_SWEEP:-0.0,0.05,0.25,0.5}"
B_SWEEP="${B_SWEEP:-0.5,0.75,0.95,1.1}"
EPS_SWEEP="${EPS_SWEEP:-0.25,0.5,1.0,2.0}"

TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

usage() {
    cat <<EOF
Usage:
  bash ${0##*/} [--skip-install]

Environment overrides:
  ISAACLAB_ROOT     IsaacLab root directory. Default: inferred from this script.
  RSL_RL_ROOT       Local rsl_rl checkout to install into IsaacLab. Default: /home/superguppy/rsl_rl
  TASKS             Comma-separated IsaacLab tasks to sweep. Default: ${DEFAULT_TASKS}
  TASK              Backward-compatible single-task override. Used only when TASKS is unset.
  AGENT             IsaacLab agent key. Default: belm_genpo
  SEEDS             Comma-separated seeds. Default: 42
  MAX_ITERATIONS    Optional global --max_iterations override. Default: use each task cfg.
  SAVE_INTERVAL     Optional global Hydra override for agent.save_interval. Default: use each task cfg.
  EXPERIMENT_PREFIX Prefix added to generated experiment_name values. Default: belm_sweep
  HEADLESS          1 to add --headless, 0 otherwise. Default: 1
  NUM_ENVS          Optional global --num_envs override. Default: use each task cfg.
  INSTALL_EDITABLE  1 to install local rsl_rl before runs, 0 otherwise. Default: 0
  A_SWEEP           Comma-separated A values. Default: 0.0,0.05,0.25,0.5
  B_SWEEP           Comma-separated B values. Default: 0.5,0.75,0.95,1.1
  EPS_SWEEP         Comma-separated eps values. Default: 0.25,0.5,1.0,2.0

Examples:
  bash ${0##*/}
  TASKS=Isaac-Velocity-Rough-H1-v0,Isaac-Humanoid-v0 SEEDS=1,2 bash ${0##*/}
  MAX_ITERATIONS=800 SAVE_INTERVAL=100 NUM_ENVS=4096 bash ${0##*/}
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
    IFS=',' read -r -a out_ref <<< "${csv}"
}

sanitize_value() {
    local value="$1"
    value="${value// /}"
    value="${value//-/neg}"
    value="${value//./p}"
    echo "${value}"
}

task_slug() {
    local task="$1"
    case "${task}" in
        Isaac-Velocity-Rough-H1-v0)
            echo "h1_rough"
            ;;
        Isaac-Velocity-Rough-G1-v0)
            echo "g1_rough"
            ;;
        Isaac-Tracking-LocoManip-Digit-v0)
            echo "digit_loco_manip"
            ;;
        Isaac-Open-Drawer-Franka-v0)
            echo "franka_open_drawer"
            ;;
        Isaac-Humanoid-v0)
            echo "humanoid"
            ;;
        *)
            echo "Unsupported task for BELM sweep: ${task}" >&2
            exit 1
            ;;
    esac
}

task_wandb_project() {
    local task="$1"
    case "${task}" in
        Isaac-Velocity-Rough-H1-v0)
            echo "belm_h1_rough"
            ;;
        Isaac-Velocity-Rough-G1-v0)
            echo "belm_g1_rough"
            ;;
        Isaac-Tracking-LocoManip-Digit-v0)
            echo "belm_digit_loco_manip"
            ;;
        Isaac-Open-Drawer-Franka-v0)
            echo "belm_franka_open_drawer"
            ;;
        Isaac-Humanoid-v0)
            echo "belm_humanoid"
            ;;
        *)
            echo "Unsupported task for BELM sweep: ${task}" >&2
            exit 1
            ;;
    esac
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

run_once() {
    local task="$1"
    local experiment_name="$2"
    local wandb_project="$3"
    local label="$4"
    shift 4

    local -a run_args=(
        --task "${task}"
        --agent "${AGENT}"
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
        "$@"
    )

    if [[ -n "${SAVE_INTERVAL}" ]]; then
        hydra_args+=("agent.save_interval=${SAVE_INTERVAL}")
    fi

    echo
    echo "==== Running ${task} :: ${label} ===="
    echo "Experiment name: ${experiment_name}"
    echo "wandb_project: ${wandb_project}"
    printf 'Overrides:'
    for arg in "${hydra_args[@]}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'

    (
        cd "${ISAACLAB_ROOT}"
        ./isaaclab.sh -p "${TRAIN_SCRIPT}" "${run_args[@]}" "${hydra_args[@]}"
    )
}

echo "IsaacLab root: ${ISAACLAB_ROOT}"
echo "rsl_rl root: ${RSL_RL_ROOT}"
echo "Tasks: ${TASKS}"
echo "Agent: ${AGENT}"
echo "Seeds: ${SEEDS}"
echo "Max iterations override: ${MAX_ITERATIONS:-<task cfg default>}"
echo "Save interval override: ${SAVE_INTERVAL:-<task cfg default>}"
echo "Experiment prefix: ${EXPERIMENT_PREFIX}"
echo "Num envs override: ${NUM_ENVS:-<task cfg default>}"
echo "A sweep: ${A_SWEEP}"
echo "B sweep: ${B_SWEEP}"
echo "eps sweep: ${EPS_SWEEP}"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
    echo
    echo "==== Installing local rsl_rl into IsaacLab Python environment ===="
    (
        cd "${ISAACLAB_ROOT}"
        ./isaaclab.sh -p -m pip install -e "${RSL_RL_ROOT}"
    )
fi

declare -a tasks
declare -a seeds
declare -a a_values
declare -a b_values
declare -a eps_values

csv_to_array "${TASKS}" tasks
csv_to_array "${SEEDS}" seeds
csv_to_array "${A_SWEEP}" a_values
csv_to_array "${B_SWEEP}" b_values
csv_to_array "${EPS_SWEEP}" eps_values

for task in "${tasks[@]}"; do
    task="${task// /}"
    task_label="$(task_slug "${task}")"
    wandb_project="$(task_wandb_project "${task}")"

    echo
    echo "==== Starting task sweep for ${task} (${task_label}) ===="

    for seed in "${seeds[@]}"; do
        seed="${seed// /}"
        baseline_variant="baseline_a$(sanitize_value "0.05")_b$(sanitize_value "0.95")_eps$(sanitize_value "1.0")"
        run_once "${task}" "$(build_experiment_name "${task_label}" "${baseline_variant}" "${seed}")" "${wandb_project}" "${baseline_variant}" \
            "agent.seed=${seed}" \
            "agent.policy.lag_coeff=null" \
            "agent.policy.a_coeff=0.05" \
            "agent.policy.b_coeff=0.95" \
            "agent.policy.eps_coeff=1.0"

        for a_value in "${a_values[@]}"; do
            a_value="${a_value// /}"
            a_variant="a_sweep_a$(sanitize_value "${a_value}")_b$(sanitize_value "0.95")_eps$(sanitize_value "1.0")"
            run_once "${task}" "$(build_experiment_name "${task_label}" "${a_variant}" "${seed}")" "${wandb_project}" "${a_variant}" \
                "agent.seed=${seed}" \
                "agent.policy.lag_coeff=null" \
                "agent.policy.a_coeff=${a_value}" \
                "agent.policy.b_coeff=0.95" \
                "agent.policy.eps_coeff=1.0"
        done

        for b_value in "${b_values[@]}"; do
            b_value="${b_value// /}"
            untied_b_variant="b_sweep_untied_a$(sanitize_value "0.05")_b$(sanitize_value "${b_value}")_eps$(sanitize_value "1.0")"
            run_once "${task}" "$(build_experiment_name "${task_label}" "${untied_b_variant}" "${seed}")" "${wandb_project}" "${untied_b_variant}" \
                "agent.seed=${seed}" \
                "agent.policy.lag_coeff=null" \
                "agent.policy.a_coeff=0.05" \
                "agent.policy.b_coeff=${b_value}" \
                "agent.policy.eps_coeff=1.0"
        done

        for eps_value in "${eps_values[@]}"; do
            eps_value="${eps_value// /}"
            eps_variant="eps_sweep_a$(sanitize_value "0.05")_b$(sanitize_value "0.95")_eps$(sanitize_value "${eps_value}")"
            run_once "${task}" "$(build_experiment_name "${task_label}" "${eps_variant}" "${seed}")" "${wandb_project}" "${eps_variant}" \
                "agent.seed=${seed}" \
                "agent.policy.lag_coeff=null" \
                "agent.policy.a_coeff=0.05" \
                "agent.policy.b_coeff=0.95" \
                "agent.policy.eps_coeff=${eps_value}"
        done

        for b_value in "${b_values[@]}"; do
            b_value="${b_value// /}"
            tied_b_variant="b_sweep_tied_lag$(sanitize_value "${b_value}")"
            run_once "${task}" "$(build_experiment_name "${task_label}" "${tied_b_variant}" "${seed}")" "${wandb_project}" "${tied_b_variant}" \
                "agent.seed=${seed}" \
                "agent.policy.a_coeff=null" \
                "agent.policy.b_coeff=null" \
                "agent.policy.eps_coeff=null" \
                "agent.policy.lag_coeff=${b_value}"
        done
    done
done

echo
echo "All BELM sweep groups finished."
