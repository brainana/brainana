#!/bin/bash

set -e  # Exit on error

# Resume is enabled by default
# Use --no-resume to start fresh
RESUME_FLAG="-resume"
if [ "$1" == "--no-resume" ] || [ "$1" == "-no-resume" ]; then
    RESUME_FLAG=""
    echo "Starting fresh: Will reprocess all steps (resume disabled)"
else
    echo "Resume mode: Will continue from previous run (use --no-resume to start fresh)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ========================
# Test parameters (edit paths/config here)
version=2.1.1

bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_devtest
output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_devtest_local_v${version}
config_f="/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/config_res-1.yaml"
# custom_template_f="/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/tpl-MEBRAINS_res-1_T1w_brain.nii.gz"
custom_template_f=""

# # newcastle
# bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_newcastle
# output_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/dataset_newcastle_sub0101
# config_f=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc/config_res-1_surf-off_anatonly.yaml

: "${ENABLE_GPU:=1}" # GPU visibility: 1 = expose GPU(s) to brainana, 0 = CPU-only (CUDA hidden)

# ========================
case "$ENABLE_GPU" in
    0|1) ;;
    *)
        echo "Error: ENABLE_GPU must be 0 or 1, got: $ENABLE_GPU" >&2
        exit 1
        ;;
esac

output_dir=${output_dir}_gpu${ENABLE_GPU}
working_dir=${output_dir}_wd

# Temp CPU config (created only in CPU mode); removed on exit.
CPU_CONFIG_F=""
cleanup_cpu_config() { [ -n "$CPU_CONFIG_F" ] && [ -f "$CPU_CONFIG_F" ] && rm -f "$CPU_CONFIG_F"; }
trap cleanup_cpu_config EXIT

# Force CPU via the pipeline's DESIGNED switch: general.gpu_device=-1 makes main.nf set
# use_gpu=false, so every GPU process receives gpu_id='none' and exports CUDA_VISIBLE_DEVICES="".
# NOTE: exporting CUDA_VISIBLE_DEVICES="" in this shell alone is NOT enough — nextflow.config still
# counts GPUs via nvidia-smi and the per-task wrapper re-exports a real GPU id, overriding it.
if [ "$ENABLE_GPU" = "0" ]; then
    export CUDA_VISIBLE_DEVICES=""   # also hides the GPU from any direct (non-wrapped) Python
    if [ -n "$config_f" ]; then
        CPU_CONFIG_F=$(mktemp /tmp/brainana_cpu_config.XXXXXX.yaml)
        "${PYTHON:-python3}" - "$config_f" "$CPU_CONFIG_F" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("general", {})["gpu_device"] = -1  # force CPU: gpu_forced_cpu -> use_gpu=false
with open(sys.argv[2], "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY
        config_f="$CPU_CONFIG_F"
        GPU_MODE="CPU-only (general.gpu_device=-1 via temp config; CUDA hidden)"
    else
        GPU_MODE="CPU-only (CUDA hidden) — WARNING: no config_f to set gpu_device=-1; CPU not guaranteed"
        echo "WARNING: ENABLE_GPU=0 but no config_f; set general.gpu_device=-1 in a config to force CPU." >&2
    fi
else
    if [ "${CUDA_VISIBLE_DEVICES-}" = "" ]; then
        unset CUDA_VISIBLE_DEVICES
        GPU_MODE="GPU enabled (all visible CUDA devices)"
    else
        GPU_MODE="GPU enabled (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    fi
fi

# Validate paths
echo "============================================"
echo "Nextflow Pipeline Test (No Docker)"
echo "============================================"
echo "BIDS directory: $bids_dir"
echo "Output directory: $output_dir"
echo "Working directory: $working_dir"
if [ -n "$config_f" ]; then
    echo "Config file: $config_f"
else
    echo "Config file: default (not overridden)"
fi
if [ -n "$custom_template_f" ]; then
    echo "Custom template: $custom_template_f"
else
    echo "Custom template: default (from config)"
fi
echo "GPU mode: $GPU_MODE (ENABLE_GPU=$ENABLE_GPU)"
echo "============================================"

# Check if BIDS directory exists
if [ ! -d "$bids_dir" ]; then
    echo "Error: BIDS directory not found: $bids_dir" >&2
    exit 1
fi

# Check config file if explicitly provided
if [ -n "$config_f" ] && [ ! -f "$config_f" ]; then
    echo "Error: Config file not found: $config_f" >&2
    exit 1
fi

# Check custom template file if explicitly provided
if [ -n "$custom_template_f" ] && [ ! -f "$custom_template_f" ]; then
    echo "Error: Custom template file not found: $custom_template_f" >&2
    exit 1
fi

# Create output and working directories if they don't exist
mkdir -p "$output_dir"
mkdir -p "$working_dir"

# Change to project root
cd "$PROJECT_ROOT"

# Run Nextflow pipeline (--no-docker flag disables Docker)
echo ""
echo "Starting Nextflow pipeline..."
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume enabled: $RESUME_FLAG"
    echo "Work directory: $working_dir"
    if [ -d "$working_dir" ] && [ -n "$(ls -A "$working_dir" 2>/dev/null)" ]; then
        echo "Work directory contains previous tasks (resume should work)"
    else
        echo "Work directory is empty (no prior run for this GPU/CPU output path)"
    fi
fi
echo ""

NF_ARGS=(
    run main.nf
    --no-docker
    --bids_dir "$bids_dir"
    --output_dir "$output_dir"
    --work_dir "$working_dir"
)
if [ -n "$config_f" ]; then
    NF_ARGS+=(--config_file "$config_f")
fi
if [ -n "$custom_template_f" ]; then
    NF_ARGS+=(--output_space "$custom_template_f")
fi
if [ -n "$RESUME_FLAG" ]; then
    NF_ARGS+=("$RESUME_FLAG")
fi
"$PROJECT_ROOT/run_brainana.sh" "${NF_ARGS[@]}"

echo ""
echo "============================================"
echo "Pipeline completed successfully!"
echo "Output directory: $output_dir"
echo "============================================"

