#!/bin/bash
# Run the dev-test dataset at each anat.synthesis_level, one level after another,
# locally (no Docker) with the repo's venv.
#
#   scripts/scratch/test_brainana_local_levels.sh                 # sub ses seslong
#   scripts/scratch/test_brainana_local_levels.sh ses seslong     # a subset
#   OUTPUT_SPACE_ses=D99:res-05 scripts/scratch/test_brainana_local_levels.sh ses
#
# Per level: copies that level's config into <root>/config.yaml (with the output space
# swapped when OUTPUT_SPACE_<level> is set), runs the pipeline into <root>/preprocessed
# with work dir <root>/preprocessed_wd, and logs to <root>/run.log.
# run_brainana.sh exits 0 even when Nextflow refuses its arguments, so success is read
# from preprocessed/nextflow_reports/run_status.json.
#
# Runs for hours: start it detached so it outlives the terminal, e.g.
#   setsid nohup scripts/scratch/test_brainana_local_levels.sh > levels.log 2>&1 < /dev/null &
set -u

version=3.1.0

bids_dir=/mnt/DataDrive3/xliu/prep_test/brainana_test/dataset_devtest
preproc=/mnt/DataDrive3/xliu/prep_test/brainana_test/preproc
out_prefix=${preproc}/dataset_devtest_local_v${version}

# Config per level (generated from src/nhp_mri_prep/config/defaults.yaml)
declare -A config_f=(
    [sub]=${preproc}/config_res-1.yaml
    [ses]=${preproc}/config_res-1_ses.yaml
    [seslong]=${preproc}/config_res-1_seslong.yaml
)

# This box caps interactive work at 4 CPUs / 50 GiB; stay inside it.
export NXF_MAX_CPUS=${NXF_MAX_CPUS:-4}
export NXF_MAX_MEMORY=${NXF_MAX_MEMORY:-40 GB}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
# The venv is required: nextflow.config picks it up from VIRTUAL_ENV.
source .venv/bin/activate

levels=("$@")
[ ${#levels[@]} -eq 0 ] && levels=(sub ses seslong)

for lvl in "${levels[@]}"; do
    src=${config_f[$lvl]:-}
    if [ -z "$src" ] || [ ! -f "$src" ]; then
        echo "=== $lvl: no config (${src:-unset}); skipped" >&2
        continue
    fi
    root=${out_prefix}_${lvl}
    mkdir -p "$root"
    cp "$src" "$root/config.yaml"
    space_var=OUTPUT_SPACE_${lvl}
    if [ -n "${!space_var:-}" ]; then
        sed -i -E "s|^(  output_space: )\"[^\"]*\"|\1\"${!space_var}\"|" "$root/config.yaml"
    fi
    echo "=== $(date '+%F %T') start $lvl ($(grep -m1 'output_space:' "$root/config.yaml" | sed -E 's/ *#.*//; s/^ *//'))"
    "$PROJECT_ROOT/run_brainana.sh" run main.nf --no-docker \
        --bids_dir "$bids_dir" \
        --output_dir "$root/preprocessed" \
        --work_dir "$root/preprocessed_wd" \
        --config_file "$root/config.yaml" > "$root/run.log" 2>&1
    status=$root/preprocessed/nextflow_reports/run_status.json
    echo "=== $(date '+%F %T') end $lvl: $(cat "$status" 2>/dev/null || echo "NO run_status.json, see $root/run.log")"
done
echo "=== ALL DONE"
