#!/usr/bin/env bash
# Preprocess PRIME-DE sites one by one with the brainana docker image.
#
#   bash test_brainana_docker_primede.sh                 # all sites below, in order
#   bash test_brainana_docker_primede.sh site-amu site-nin   # only these
#
# Re-runnable: sites with <out_root>/<site>/.done are skipped; an interrupted site
# resumes from its work dir (<out_root>/_wd/<site>). A failed site does not stop the batch.
# Per-site log: <out_root>/_logs/<site>.log; one row per run in <out_root>/_logs/summary.tsv.
set -u

fs_license=/mnt/DataDrive3/xliu/prep_test/freesurfer_license.txt
version=3.2.0
image=liuxingyu987/brainana:${version}

raw_root=/mnt/DataDrive2/macaque/data_raw/macaque_mri/PRIME-DE
out_root=/mnt/DataDrive2/macaque/data_preproc/macaque_mri/PRIME-DE_brainana_v${version}

# docker runs outside the user slice, so these are the only limits (shared box: <= 40g)
cpus=16
mem=40g
nxf_mem=36g

# smallest -> largest
sites=(
    site-ecnuChen site-ecnu site-rochester site-mcgill site-amu site-lyne site-ion
    site-sbri site-ds003989 site-uminn site-caltech site-princeton site-newcastle site-nin
    site-mountsinaiP site-oxford site-mountsinaiS site-carmenlyon site-rockefeller site-ohsu
    site-neurospin site-bordeaux site-nki site-uwo site-iscmj site-ucdavis
)
[ $# -gt 0 ] && sites=("$@")

# ------------------------------------------------------------
# preflight
[ -f "$fs_license" ] || { echo "ERROR: license not found: $fs_license" >&2; exit 1; }
docker image inspect "$image" >/dev/null 2>&1 || { echo "ERROR: image not found: $image" >&2; exit 1; }
for site in "${sites[@]}"; do
    [ -d "$raw_root/$site" ] || { echo "ERROR: raw site not found: $raw_root/$site" >&2; exit 1; }
done

log_dir=$out_root/_logs
summary=$log_dir/summary.tsv
mkdir -p "$log_dir"
[ -f "$summary" ] || printf 'date\tsite\trc\tfailed_tasks\tduration_min\tstatus\n' > "$summary"

# Tasks whose LAST attempt is FAILED/ABORTED (retried-then-succeeded tasks don't count).
# Prints -1 if the trace is missing.
count_failed_tasks() {
    local trace=$1
    [ -f "$trace" ] || { echo -1; return; }
    awk -F'\t' 'NR > 1 { last[$4] = $5 }
        END { n = 0; for (k in last) if (last[k] == "FAILED" || last[k] == "ABORTED") n++; print n }' "$trace"
}

# ------------------------------------------------------------
for site in "${sites[@]}"; do
    output_dir=$out_root/$site
    work_dir=$out_root/_wd/$site

    if [ -f "$output_dir/.done" ]; then
        echo "=== $site: already done, skipping"
        printf '%s\t%s\t-\t-\t-\tSKIP\n' "$(date '+%F %T')" "$site" >> "$summary"
        continue
    fi

    # created as the host user, so the container drops to this uid (outputs not root-owned)
    mkdir -p "$output_dir" "$work_dir"

    echo "=== $site: start $(date '+%F %T')"
    start=$(date +%s)

    docker run --rm -t --gpus all --cpus "$cpus" --memory "$mem" \
        --name "brainana_primede_${site}" \
        -e NXF_MAX_CPUS="$cpus" -e NXF_MAX_MEMORY="$nxf_mem" \
        -v "$raw_root/$site":/input:ro \
        -v "$output_dir":/output \
        -v "$work_dir":/work \
        -v "$fs_license":/fs_license.txt:ro \
        "$image" \
        /input /output \
        -w /work \
        --freesurfer_license /fs_license.txt \
        2>&1 | tee "$log_dir/${site}.log"
    rc=${PIPESTATUS[0]}

    dur=$(( ($(date +%s) - start) / 60 ))
    failed=$(count_failed_tasks "$output_dir/nextflow_reports/nextflow_trace.txt")

    if [ "$rc" -eq 0 ] && [ "$failed" -eq 0 ]; then
        status=OK
        touch "$output_dir/.done"
    else
        status=FAIL
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$site" "$rc" "$failed" "$dur" "$status" >> "$summary"
    echo "=== $site: $status (rc=$rc, failed_tasks=$failed, ${dur} min)"
done

echo "=== batch finished; summary: $summary"
