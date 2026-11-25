#!/bin/bash
#==============================================================================
# Train a segmentation CNN view (or several) from YAML configs.
#
#   run_training_pipeline.sh --config CFG.yaml [--config CFG2.yaml ...]
#                            [--steps split,hdf5,train,eval] [--gpu N]
#                            [--profile N] [--resume] [-- KEY VALUE ...]
#
# Steps (default: hdf5,train):
#   split  step1_split_data.py   animal-level train/val split -> TRAINING_DATA_DIR/data_split.json
#                                (skipped if the file exists; monkey_unet_prep/5_balance_sites.py
#                                usually writes it)
#   hdf5   step2_create_hdf5.py  train + val HDF5 for the config's plane (resumable)
#   train  step3_train_model.py  fine-tune / train; progress in OUTPUT_DIR/monitor/
#   eval   evaluate_checkpoint.py per-subject Dice of the run's best checkpoint on val
#
# Configs run one after another (one GPU job at a time). Stops at the first failure.
# Watch progress live:
#   python -m fastsurfer_nn.scripts.training_dashboard --runs <OUTPUT_DIR> ...
# Anything after "--" is passed to step3 as config overrides, e.g. -- TRAIN.NUM_EPOCHS 5
#==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SRC="$PROJECT_ROOT/src/fastsurfer_nn"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIGS=()
STEPS="hdf5,train"
PROFILE=0
RESUME=""
OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIGS+=("$2"); shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --resume) RESUME="--resume"; shift ;;
        --) shift; OVERRIDES=("$@"); break ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "give at least one --config" >&2; exit 2; }
has_step() { [[ ",$STEPS," == *",$1,"* ]]; }

yaml_get() {  # yaml_get FILE KEY  (top-level key)
    python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"
}

for CFG in "${CONFIGS[@]}"; do
    echo "=============================================================================="
    echo "config: $CFG   steps: $STEPS   GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
    echo "=============================================================================="
    DATA_DIR="$(yaml_get "$CFG" TRAINING_DATA_DIR)"
    OUT_DIR="$(yaml_get "$CFG" OUTPUT_DIR)"

    if has_step split; then
        if [[ -f "$DATA_DIR/data_split.json" ]]; then
            echo "[split] $DATA_DIR/data_split.json exists - keeping it"
        else
            python3 "$SRC/training/step1_split_data.py" --config "$CFG"
        fi
    fi
    if has_step hdf5; then
        echo "[hdf5] train"
        python3 "$SRC/training/step2_create_hdf5.py" --config "$CFG" --split_type train
        echo "[hdf5] val"
        python3 "$SRC/training/step2_create_hdf5.py" --config "$CFG" --split_type val
    fi
    if has_step train; then
        echo "[train] output: $OUT_DIR"
        echo "        dashboard: python -m fastsurfer_nn.scripts.training_dashboard --runs $OUT_DIR"
        PROF_ARGS=()
        [[ "$PROFILE" != "0" ]] && PROF_ARGS=(--profile "$PROFILE")
        python3 "$SRC/training/step3_train_model.py" --cfg "$CFG" $RESUME "${PROF_ARGS[@]}" "${OVERRIDES[@]}"
    fi
    if has_step eval; then
        echo "[eval] $OUT_DIR/checkpoints/Best_training_state.pkl"
        python3 -m fastsurfer_nn.scripts.evaluate_checkpoint \
            --checkpoints "$OUT_DIR/checkpoints/Best_training_state.pkl" \
            --data-dir "$DATA_DIR" --split val --out "$OUT_DIR/eval"
    fi
done
echo "done"
