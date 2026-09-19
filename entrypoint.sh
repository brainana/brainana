#!/bin/bash
#
# brainana Docker entrypoint
#
# Starts as root for setup (GPU/CUDA works for any UID with nvidia-container-toolkit).
# When the mounted output dir is owned by a non-root UID, the pipeline is re-exec'd via
# gosu as that UID:GID so derivatives are not root-owned on the host.
#
# Usage (production):
#   docker run --rm --gpus all \
#     -v <path/to/bids_dir>:/input \
#     -v <path/to/output_dir>:/output \
#     -v <path/to/work_dir>:/output_wd \
#     -v <path/to/license.txt>:/fs_license.txt \
#     liuxingyu987/brainana:<version> /input /output \
#     --work_dir /output_wd --freesurfer_license /fs_license.txt
#
# (Connectors: underscore is canonical; the hyphenated spellings --work-dir,
#  --freesurfer-license, --no-resume are still accepted as aliases.)
#
# With custom config:
#   docker run ... liuxingyu987/brainana:<version> /input /output --work_dir /output_wd --config /config.yaml ...
#
# With a custom template (mount the .nii/.nii.gz and pass the in-container path;
# outputs are labelled space-template):
#   docker run ... -v <path/to/template.nii.gz>:/template.nii.gz \
#     liuxingyu987/brainana:<version> /input /output --work_dir /output_wd --output_space /template.nii.gz
#
# For interactive shell:
#   docker run -it ... liuxingyu987/brainana:<version> bash
#

set -e

# Source neuroimaging env (PATH, LD_LIBRARY_PATH, LD_PRELOAD for FireANTs fused CUDA ops, etc.)
# neuroenv.sh is normally only sourced for interactive bash shells via /etc/bash.bashrc.
# Sourcing it here explicitly ensures these variables are exported and inherited by gosu
# and all Nextflow task subprocesses, which run non-interactively.
# shellcheck disable=SC1091
[ -f /etc/profile.d/neuroenv.sh ] && source /etc/profile.d/neuroenv.sh

# --- functions ----------------------------------------------------------------

freesurfer_license_probe() {
    local probe_input="${FREESURFER_HOME:-/usr/local/freesurfer}/average/pons.mni152.2mm.mgz"
    local probe_output="/tmp/brainana_fs_license_test.nii.gz"
    local probe_log="/tmp/brainana_fs_license_probe.log"

    if [ ! -f "$probe_input" ]; then
        return 0
    fi

    if ! mri_convert "$probe_input" "$probe_output" >"$probe_log" 2>&1; then
        echo "ERROR: FreeSurfer license check failed for FS_LICENSE=$FS_LICENSE" >&2
        if [ -s "$probe_log" ]; then
            cat "$probe_log" >&2
        fi
        rm -f "$probe_output" "$probe_log"
        exit 1
    fi

    rm -f "$probe_output" "$probe_log"
}

# Root only. Sets globals: PIPELINE_UID, PIPELINE_GID, GOSU_TARGET, PIPELINE_MODE.
# PIPELINE_MODE: gosu | root_fallback | root_output_root_owned
setup_pipeline_identity() {
    local output_dir="$1"
    local work_dir="$2"
    local _pu_name _run_as

    PIPELINE_UID=$(stat -c %u "$output_dir" 2>/dev/null || echo 0)
    PIPELINE_GID=$(stat -c %g "$output_dir" 2>/dev/null || echo 0)
    GOSU_TARGET=""
    PIPELINE_MODE="root_output_root_owned"

    if [ "$PIPELINE_UID" = "0" ]; then
        PIPELINE_UID=""
        PIPELINE_GID=""
        return
    fi

    # This ran unconditionally on every start. A resumed run's work tree is tens of GB
    # and can be millions of inodes, so `chown -R` spent minutes doing nothing before
    # the pipeline printed anything. Skip it when the tree root already carries the
    # target ownership, which holds for every resume after the first launch.
    # Caveat: a tree left half-owned by an interrupted chown is not repaired here --
    # delete the work dir to force a clean pass.
    local work_dir_owner
    work_dir_owner="$(stat -c '%u:%g' "$work_dir" 2>/dev/null || echo '')"
    if [ "$work_dir_owner" = "${PIPELINE_UID}:${PIPELINE_GID}" ]; then
        : # already owned by the pipeline user
    elif ! chown -R "${PIPELINE_UID}:${PIPELINE_GID}" "$work_dir" 2>/dev/null; then
        echo "WARNING: Could not chown work dir $work_dir to ${PIPELINE_UID}:${PIPELINE_GID}" >&2
        echo "         (e.g. NFS root_squash); work files may stay root-owned." >&2
    fi

    if ! getent passwd "$PIPELINE_UID" >/dev/null 2>&1; then
        if ! getent group "$PIPELINE_GID" >/dev/null 2>&1; then
            groupadd -g "$PIPELINE_GID" "brainana-g${PIPELINE_GID}" 2>/dev/null || true
        fi
        _pu_name="brainana${PIPELINE_UID}"
        if ! useradd -M -u "$PIPELINE_UID" -g "$PIPELINE_GID" -d /tmp/home -s /bin/bash "$_pu_name" 2>/dev/null; then
            echo "WARNING: Could not add /etc/passwd entry for UID $PIPELINE_UID; running pipeline as root instead." >&2
            echo "         New outputs will be root-owned. On the host after the run, fix with e.g." >&2
            echo "         sudo chown -R \"\$(id -u):\$(id -g)\" <mounted output and work paths>" >&2
            PIPELINE_MODE="root_fallback"
            return
        fi
    fi

    # Grant GPU device access to the pipeline user.
    # nvidia-container-toolkit mounts /dev/nvidia* from the host preserving the
    # host group ownership (e.g. GID 1023 = vglusers on this cluster). The
    # container has no group with that GID by default, so we create one and add
    # the user. This is the portable fix regardless of the group name on the host.
    _pu_name="$(getent passwd "$PIPELINE_UID" | cut -d: -f1)"
    _nvidia_gid=$(stat -c %g /dev/nvidiactl 2>/dev/null \
                  || stat -c %g /dev/nvidia0 2>/dev/null \
                  || echo "")
    if [ -n "$_nvidia_gid" ] && [ "$_nvidia_gid" != "0" ]; then
        if ! getent group "$_nvidia_gid" >/dev/null 2>&1; then
            groupadd -g "$_nvidia_gid" "nvidia-${_nvidia_gid}" 2>/dev/null || true
        fi
        _nvidia_grp=$(getent group "$_nvidia_gid" | cut -d: -f1)
        [ -n "$_nvidia_grp" ] && usermod -aG "$_nvidia_grp" "$_pu_name" 2>/dev/null || true
    fi
    # Also add to render for /dev/dri/* (DRM/OpenGL devices used by some CUDA ops).
    for _grp in render; do
        if getent group "$_grp" >/dev/null 2>&1; then
            usermod -aG "$_grp" "$_pu_name" 2>/dev/null || true
        fi
    done

    # Use the username (not numeric uid:gid) so gosu calls initgroups(), which
    # loads ALL supplementary groups from /etc/group — including the nvidia-GID
    # group added above. With numeric "uid:gid", gosu skips initgroups() and the
    # supplementary groups are never applied, leaving GPU devices inaccessible.
    _run_as="$(getent passwd "$PIPELINE_UID" | cut -d: -f1)"
    GOSU_TARGET="$_run_as"
    PIPELINE_MODE="gosu"
    export HOME="/tmp/home"
    mkdir -p "$HOME" 2>/dev/null || true
    export NXF_TEMP=/tmp
    if [ -n "$_run_as" ]; then
        export USER="$_run_as"
        export LOGNAME="$_run_as"
    fi
}

print_pipeline_banner() {
    local resume_on="$1"
    local gpu="$2"
    local rs
    rs=$([ "$resume_on" -eq 1 ] && echo on || echo off)

    echo "============================================"
    echo "brainana pipeline"
    echo "  Input:   $INPUT_DIR"
    echo "  Output:  $OUTPUT_DIR"
    echo "  Config:  $CONFIG"
    echo "  Work:    $NXF_HOME (resume: $rs)"
    case "$PIPELINE_MODE" in
        gosu)
            echo "  User:    $GOSU_TARGET (matches mounted output dir owner)"
            ;;
        root_fallback)
            echo "  User:    0:0 (root fallback — useradd failed for host UID $PIPELINE_UID; new files root-owned)"
            echo "  Hint:    On the host after the run: sudo chown -R \"\$(id -u):\$(id -g)\" your mounted output/work paths."
            ;;
        root_output_root_owned)
            echo "  User:    0:0 (pipeline as root; new outputs stay root-owned)"
            echo "  Hint:    To run as your user: on the host, pre-create or chown the mounted output and work dirs, then docker run again."
            ;;
        explicit_u)
            echo "  User:    $(id -u):$(id -g) (docker run -u)"
            ;;
    esac
    echo "  GPUs:    $gpu"
    echo "============================================"
}

# --- main ---------------------------------------------------------------------

PIPELINE_MODE=""
PIPELINE_UID=""
PIPELINE_GID=""
GOSU_TARGET=""

if [ "$(id -u)" != "0" ]; then
    export HOME="/tmp/home"
    mkdir -p "$HOME" 2>/dev/null || true
    export NXF_TEMP=/tmp
    PIPELINE_MODE="explicit_u"
fi

export NXF_NO_DOCKER=1
export NXF_ANSI_LOG="${NXF_ANSI_LOG:-true}"
export NXF_MAX_CPUS="${NXF_MAX_CPUS:-8}"
export NXF_MAX_MEMORY="${NXF_MAX_MEMORY:-20g}"

DEFAULT_CONFIG="/opt/brainana/src/nhp_mri_prep/config/defaults.yaml"
PROJECT_ROOT="/opt/brainana"

# Shared CLI-flag helpers: KNOWN_FLAGS (from known_flags.txt), normalize_flag,
# validate_flags, print_usage. Single source of truth shared with run_brainana.sh + main.nf.
# Guarded so a missing flags.sh degrades gracefully instead of aborting under `set -e`.
# shellcheck source=flags.sh
if [ -f "$PROJECT_ROOT/flags.sh" ]; then
    . "$PROJECT_ROOT/flags.sh"
else
    echo "WARNING: flags.sh not found; --help and unknown-argument validation are disabled." >&2
    validate_flags() { UNKNOWN_FLAGS=(); return 0; }
    normalize_flag() { printf '%s' "$1"; }
    print_usage() { echo "brainana — see docs/usage_notes.rst"; }
fi

# Show usage for -h/--help appearing anywhere in the arguments, before any heavy setup
# (GPU/Xvfb/Nextflow). Handles both `image --help` and `image /in /out --help`. Skipped
# when the first arg requests an interactive shell (bash/sh), handled further below.
case "${1:-}" in
    bash|sh|-bash|-sh) ;;
    *)
        for _a in "$@"; do
            case "$_a" in
                -h|--help) print_usage; exit 0 ;;
            esac
        done
        ;;
esac

# Positionals are optional, and must not swallow a leading flag. CMD supplies
# "/input /output", but ANY user argument replaces CMD wholesale, so
# `docker run <image> --anat_only` arrives as $1="--anat_only". That used to be
# taken as the input directory and rejected with the misleading
# "Input directory not found: --anat_only". Only a non-flag token is consumed as a
# positional; "bash"/"sh" still land in INPUT_DIR and are intercepted further below.
INPUT_DIR="/input"
OUTPUT_DIR="/output"
i=1
if [ $# -ge $i ] && [[ "${!i}" != -* ]]; then
    INPUT_DIR="${!i}"
    i=$((i + 1))
    if [ $# -ge $i ] && [[ "${!i}" != -* ]]; then
        OUTPUT_DIR="${!i}"
        i=$((i + 1))
    fi
fi

CONFIG="$DEFAULT_CONFIG"
WORK_DIR=""
RESUME_BY_DEFAULT=1
EXTRA_ARGS=()
FS_LICENSE_PATH=""
while [ $i -le $# ]; do
    # Underscore is the canonical connector. normalize_flag() (flags.sh) maps the
    # hyphenated aliases onto it for every name in known_flags.txt, so only canonical
    # spellings need matching below. Both spellings used to be hardcoded here AND in
    # run_brainana.sh, so the two alias tables could drift apart. If known_flags.txt is
    # missing, flags.sh already warns and normalization degrades to a pass-through.
    arg="$(normalize_flag "${!i}")"
    if [[ "$arg" == --config=* ]] || [[ "$arg" == --config_file=* ]]; then
        CONFIG="${arg#*=}"
    elif [[ "$arg" == --config ]] || [[ "$arg" == --config_file ]]; then
        ((i++))
        [ $i -le $# ] && CONFIG="${!i}"
    elif [[ "$arg" == --work_dir=* ]]; then
        WORK_DIR="${arg#*=}"
    elif [[ "$arg" == -w ]] || [[ "$arg" == --work_dir ]]; then
        ((i++))
        [ $i -le $# ] && WORK_DIR="${!i}"
    elif [[ "$arg" == --no_resume ]]; then
        RESUME_BY_DEFAULT=0
    elif [[ "$arg" == --freesurfer_license=* ]]; then
        # Consumed here into FS_LICENSE (below); NOT forwarded to Nextflow, which has
        # no freesurfer_license param — forwarding it would only create an unused param.
        FS_LICENSE_PATH="${arg#*=}"
    elif [[ "$arg" == --freesurfer_license ]]; then
        ((i++))
        [ $i -le $# ] && FS_LICENSE_PATH="${!i}"
    else
        EXTRA_ARGS+=("$arg")
    fi
    ((i++))
done

if [ -n "$FS_LICENSE_PATH" ]; then
    export FS_LICENSE="$FS_LICENSE_PATH"
else
    unset FS_LICENSE
fi

# Fail fast on unrecognized arguments BEFORE any heavy setup (FS-license probe, GPU
# detection, Xvfb, Nextflow). EXTRA_ARGS holds the flags this entrypoint did not consume;
# a typo like --output_space__ is rejected here in ~1s. run_brainana.sh + main.nf re-check.
if [ ${#EXTRA_ARGS[@]} -gt 0 ] && ! validate_flags "${EXTRA_ARGS[@]}"; then
    echo "Unknown argument(s): ${UNKNOWN_FLAGS[*]}" >&2
    echo "" >&2
    print_usage >&2
    echo "" >&2
    echo "Hint: custom templates use --output_space <file>, not --custom-template." >&2
    exit 2
fi

if [ $# -gt 0 ]; then
    case "$1" in
        bash|sh|-bash|-sh)
            exec "$@"
            exit 0
            ;;
    esac
fi

# --- validate I/O -------------------------------------------------------------

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR" >&2
    echo "Usage: docker run ... liuxingyu987/brainana:<version> [input_dir] [output_dir]" >&2
    echo "       Default: /input /output (must be mounted with -v)" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR" || {
        echo "ERROR: Cannot create output directory: $OUTPUT_DIR" >&2
        echo "Check that the parent mount is writable." >&2
        exit 1
    }
fi
if ! [ -w "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory is not writable: $OUTPUT_DIR" >&2
    echo "Create it on the host with correct ownership before docker run, or fix permissions." >&2
    echo "If you use docker run -u UID:GID, that UID must be able to write this path." >&2
    exit 1
fi

# --- work dir + pipeline identity ---------------------------------------------

if [ -z "$WORK_DIR" ]; then
    WORK_DIR="${OUTPUT_DIR}_wd"
fi
export NXF_HOME="$WORK_DIR"
export NXF_WORK="${WORK_DIR}/work"
export NXF_LAUNCH_DIR="$WORK_DIR"
mkdir -p "$NXF_HOME" "$NXF_WORK" 2>/dev/null || true
if [ -d /opt/nextflow/framework ] && [ ! -e "$NXF_HOME/framework" ]; then
    ln -s /opt/nextflow/framework "$NXF_HOME/framework" 2>/dev/null || true
fi

if [ "$(id -u)" = "0" ]; then
    setup_pipeline_identity "$OUTPUT_DIR" "$WORK_DIR"
fi

if [ "$RESUME_BY_DEFAULT" -eq 1 ]; then
    EXTRA_ARGS+=("-resume")
fi

SURF_RECON_ENABLED="$(
python3 - "$CONFIG" "$DEFAULT_CONFIG" << 'PY'
import io
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    print("true")
    sys.exit(0)

user_cfg_path = Path(sys.argv[1])
default_cfg_path = Path(sys.argv[2])

def load_yaml(p: Path):
    if not p.is_file():
        return {}
    try:
        raw = p.read_text()
        raw = raw.replace("\t", " " * 4)
        return yaml.safe_load(io.StringIO(raw)) or {}
    except Exception:
        return {}

cfg = load_yaml(default_cfg_path)
user_cfg = load_yaml(user_cfg_path)
anat = cfg.get("anat") or {}
user_anat = user_cfg.get("anat") or {}
surf = anat.get("surface_reconstruction") or {}
user_surf = user_anat.get("surface_reconstruction") or {}
enabled = surf.get("enabled", True)
if "enabled" in user_surf:
    enabled = bool(user_surf["enabled"])
print("true" if enabled else "false")
PY
)"

if [ "$SURF_RECON_ENABLED" = "true" ]; then
    if [ -z "$FS_LICENSE_PATH" ]; then
        echo "ERROR: Surface reconstruction is enabled but --freesurfer_license was not provided." >&2
        echo "       Pass --freesurfer_license /path/to/license.txt (mount with -v /host/license.txt:/path/to/license.txt)" >&2
        exit 1
    fi
    if [ ! -f "$FS_LICENSE" ]; then
        echo "ERROR: Surface reconstruction is enabled but FreeSurfer license file was not found at $FS_LICENSE." >&2
        echo "       Ensure the license file is mounted and the path matches --freesurfer_license." >&2
        exit 1
    fi
    freesurfer_license_probe
fi

cd "$PROJECT_ROOT"

GPU_COUNT=0
if command -v nvidia-smi &>/dev/null && nvidia-smi --list-gpus &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
fi

# When no GPU is accessible inside the container (e.g. docker run without --gpus),
# hide all CUDA devices so every tool (FastSurfer, FireANTs, skullstripping) sees
# no GPU and falls back to CPU immediately — without attempting broken CUDA ops.
# Only set when not already overridden by the user.
if [ "$GPU_COUNT" -eq 0 ] && [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

print_pipeline_banner "$RESUME_BY_DEFAULT" "$GPU_COUNT"

export DISPLAY="${DISPLAY:-:99}"
Xvfb "$DISPLAY" -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 1
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "WARNING: Xvfb failed to start; QC snapshots may fail." >&2
fi

set -- ./run_brainana.sh run main.nf \
    --bids_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_file "$CONFIG" \
    "${EXTRA_ARGS[@]}"

if [ -n "$GOSU_TARGET" ]; then
    if ! command -v gosu &>/dev/null; then
        echo "ERROR: gosu not found; cannot drop to $GOSU_TARGET" >&2
        exit 1
    fi
    exec gosu "$GOSU_TARGET" "$@"
fi
exec "$@"
