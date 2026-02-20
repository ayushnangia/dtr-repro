#!/bin/bash
# Submit SLURM jobs for DTR experiments on Alliance Canada Killarney.
#
# IMPORTANT: Before first use, replace 'def-CHANGEME' with your actual
# allocation account in all .sbatch files, or set the ACCOUNT env var:
#   export ACCOUNT=def-yourpi
#
# Usage:
#   # Submit all Table 1 jobs (2 models x 4 benchmarks)
#   ./slurm/submit_all.sh --experiment table1
#
#   # Submit Table 2 jobs (Qwen3-4B x 4 benchmarks, 48 samples)
#   ./slurm/submit_all.sh --experiment table2
#
#   # Submit all jobs with custom seeds
#   ./slurm/submit_all.sh --experiment table1 --seeds "42 123 456"
#
#   # Submit specific model/benchmark combination
#   ./slurm/submit_all.sh --experiment custom --models "deepseek_r1_70b" --benchmarks "aime_2025"
#
#   # Dry run (print commands without submitting)
#   ./slurm/submit_all.sh --experiment table1 --dry_run

set -euo pipefail

# Defaults
EXPERIMENT=""
SEEDS="42"
N_SAMPLES=25
DRY_RUN=0
STORE_JSD=0
CUSTOM_MODELS=""
CUSTOM_BENCHMARKS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --models)
            CUSTOM_MODELS="$2"
            shift 2
            ;;
        --benchmarks)
            CUSTOM_BENCHMARKS="$2"
            shift 2
            ;;
        --store_jsd)
            STORE_JSD=1
            shift
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --experiment EXPERIMENT [OPTIONS]"
            echo ""
            echo "Submit DTR generation jobs on Alliance Canada Killarney (H100 GPUs)."
            echo ""
            echo "Options:"
            echo "  --experiment NAME    Experiment to run: table1, table2, table3, figure4, appendix_a, appendix_c, custom"
            echo "  --seeds SEEDS        Space-separated list of base seeds (default: '42')"
            echo "  --n_samples N        Number of samples per question (default: 25)"
            echo "  --models MODELS      Space-separated model names (for --experiment custom)"
            echo "  --benchmarks BENCH   Space-separated benchmark names (for --experiment custom)"
            echo "  --store_jsd          Save full JSD matrices (needed for figure4, appendix_a)"
            echo "  --dry_run            Print commands without submitting"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Experiments:"
            echo "  table1       Correlation analysis: deepseek_r1_70b, qwen3_30b x 4 benchmarks"
            echo "  table2       Think@n comparison: qwen3_4b x 4 benchmarks, n_samples=48"
            echo "  table3       Prefix ablation: qwen3_4b x aime_2025, n_samples=48"
            echo "  figure4      Sensitivity sweep (reuses table1 data, needs --store_jsd)"
            echo "  appendix_a   Distance metrics (reuses table1 data, needs --store_jsd)"
            echo "  appendix_c   Think@n analysis: qwen3_4b x 4 benchmarks, n_samples=48"
            echo "  custom       Specify --models and --benchmarks manually"
            echo ""
            echo "Cluster: Killarney (H100-80GB GPUs)"
            echo "Environment: ~/dtr-env (run slurm/setup_env.sh first)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${EXPERIMENT}" ]]; then
    echo "ERROR: --experiment is required. Use --help for options."
    exit 1
fi

# Create logs directory
mkdir -p slurm/logs

# Helper function to submit a job
submit_job() {
    local MODEL="$1"
    local BENCHMARK="$2"
    local SEED="$3"
    local SBATCH_FILE="$4"
    local EXTRA_N_SAMPLES="${5:-${N_SAMPLES}}"

    # Override account if ACCOUNT env var is set
    local ACCOUNT_FLAG=""
    if [[ -n "${ACCOUNT:-}" ]]; then
        ACCOUNT_FLAG="--account=${ACCOUNT}"
    fi

    local CMD="BENCHMARK=${BENCHMARK} N_SAMPLES=${EXTRA_N_SAMPLES} SEED=${SEED} STORE_JSD=${STORE_JSD} sbatch ${ACCOUNT_FLAG} ${SBATCH_FILE}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "[DRY RUN] ${CMD}"
    else
        echo "Submitting: ${CMD}"
        JOB_ID=$(eval ${CMD} | awk '{print $NF}')
        echo "  -> Job ID: ${JOB_ID}"
    fi
}

# Map model names to sbatch files
get_sbatch_file() {
    local MODEL="$1"
    case "${MODEL}" in
        deepseek_r1_70b) echo "slurm/generate_deepseek_70b.sbatch" ;;
        qwen3_30b)       echo "slurm/generate_qwen3_30b.sbatch" ;;
        qwen3_4b)        echo "slurm/generate_qwen3_4b.sbatch" ;;
        *)
            echo "ERROR: Unknown model ${MODEL}" >&2
            exit 1
            ;;
    esac
}

# Define experiment configurations
case "${EXPERIMENT}" in
    table1)
        MODELS="deepseek_r1_70b qwen3_30b"
        BENCHMARKS="aime_2025 hmmt_2025 gpqa_diamond"
        N_SAMPLES=25
        ;;
    table2)
        MODELS="qwen3_4b"
        BENCHMARKS="aime_2025 hmmt_2025 gpqa_diamond"
        N_SAMPLES=48
        ;;
    table3)
        MODELS="qwen3_4b"
        BENCHMARKS="aime_2025"
        N_SAMPLES=48
        ;;
    figure4)
        echo "Figure 4 reuses Table 1 data. Ensure Table 1 was run with --store_jsd."
        echo "Run: python scripts/compute_metrics.py to sweep hyperparameters."
        if [[ ${STORE_JSD} -eq 0 ]]; then
            echo "WARNING: --store_jsd not set. If Table 1 was not run with JSD storage,"
            echo "         re-run Table 1 with --store_jsd first."
        fi
        MODELS="deepseek_r1_70b qwen3_30b"
        BENCHMARKS="gpqa_diamond"
        STORE_JSD=1
        ;;
    appendix_a)
        echo "Appendix A reuses Table 1 data with JSD matrices."
        MODELS="deepseek_r1_70b qwen3_30b"
        BENCHMARKS="aime_2025 hmmt_2025"
        STORE_JSD=1
        ;;
    appendix_c)
        MODELS="qwen3_4b"
        BENCHMARKS="aime_2025 hmmt_2025 gpqa_diamond"
        N_SAMPLES=48
        ;;
    custom)
        if [[ -z "${CUSTOM_MODELS}" ]]; then
            echo "ERROR: --models required for --experiment custom"
            exit 1
        fi
        if [[ -z "${CUSTOM_BENCHMARKS}" ]]; then
            echo "ERROR: --benchmarks required for --experiment custom"
            exit 1
        fi
        MODELS="${CUSTOM_MODELS}"
        BENCHMARKS="${CUSTOM_BENCHMARKS}"
        ;;
    *)
        echo "ERROR: Unknown experiment '${EXPERIMENT}'. Use --help for options."
        exit 1
        ;;
esac

echo "=============================================="
echo "DTR Experiment Submission (Killarney H100)"
echo "=============================================="
echo "Experiment:  ${EXPERIMENT}"
echo "Models:      ${MODELS}"
echo "Benchmarks:  ${BENCHMARKS}"
echo "N Samples:   ${N_SAMPLES}"
echo "Seeds:       ${SEEDS}"
echo "Store JSD:   ${STORE_JSD}"
echo "Dry Run:     ${DRY_RUN}"
echo "=============================================="
echo ""

# Submit jobs
JOB_COUNT=0
for SEED in ${SEEDS}; do
    for MODEL in ${MODELS}; do
        SBATCH_FILE=$(get_sbatch_file "${MODEL}")
        for BENCHMARK in ${BENCHMARKS}; do
            submit_job "${MODEL}" "${BENCHMARK}" "${SEED}" "${SBATCH_FILE}" "${N_SAMPLES}"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "=============================================="
echo "Submitted ${JOB_COUNT} array jobs."
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "(Dry run -- no jobs were actually submitted)"
fi
echo ""
echo "Monitor with: sq"
echo "Check logs:   tail -f slurm/logs/*.out"
echo "=============================================="
