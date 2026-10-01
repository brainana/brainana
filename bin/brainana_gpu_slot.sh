# GPU slot for a Nextflow task. Sourced, not executed, by the script of a process
# that runs on the GPU:
#
#     source brainana_gpu_slot.sh     # bin/ is on the task PATH
#
# Takes a free slot, holds it until the task exits, and exports
# CUDA_VISIBLE_DEVICES=<gpu> and BRAINANA_DEVICE=cuda.
#
# The slot table comes from the global beforeScript in nextflow.config:
#   BRAINANA_GPU_SLOTS     physical GPU id of each slot, space separated ("0 0 1 1").
#                          Empty in CPU mode: the task then stays on the CPU.
#   BRAINANA_GPU_LOCK_DIR  directory for the slot lock files, under the work dir.
#
# Why a lock and not a token passed between processes:
#   - The lock is an flock on a descriptor this shell keeps open; its child
#     processes inherit it. The kernel releases it when they exit, however they exit,
#     so a failed, killed or 'ignore'd task cannot leak a slot.
#   - The GPU id never appears in the task script or its inputs, so it is not part
#     of the Nextflow task hash: -resume does not depend on which GPU a task drew.

brainana_acquire_gpu_slot() {
    local slots="${BRAINANA_GPU_SLOTS:-}"
    local dir="${BRAINANA_GPU_LOCK_DIR:-}"
    if [ -z "$slots" ] || [ -z "$dir" ]; then
        echo "[GPU Assignment] no GPU slots (CPU mode)"
        return 0
    fi
    mkdir -p "$dir"
    local waited=0 i gpu fd rc
    while true; do
        i=0
        for gpu in $slots; do
            exec {fd}>>"$dir/slot${i}_gpu${gpu}.lock"
            # 75 = held by another task. Anything else is an error (e.g. a work dir
            # on Lustre or NFS without lock support), which would otherwise look
            # like a busy slot forever.
            rc=0
            flock -n -E 75 "$fd" || rc=$?
            if [ "$rc" -eq 0 ]; then
                export CUDA_VISIBLE_DEVICES="$gpu" BRAINANA_DEVICE=cuda
                if [ "$waited" -gt 0 ]; then
                    echo "[GPU Assignment] slot $i -> GPU $gpu (waited ${waited}s)"
                else
                    echo "[GPU Assignment] slot $i -> GPU $gpu"
                fi
                return 0
            fi
            exec {fd}>&-
            if [ "$rc" -ne 75 ]; then
                echo "[GPU Assignment] ERROR: cannot lock $dir/slot${i}_gpu${gpu}.lock" \
                    "(flock exit $rc). The work dir's file system may not support" \
                    "file locks (Lustre/NFS): use a work dir on local disk." >&2
                exit 1
            fi
            i=$((i + 1))
        done
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 600)) -eq 0 ]; then
            echo "[GPU Assignment] waiting for a GPU slot (${waited}s)"
        fi
    done
}

brainana_acquire_gpu_slot
