#!/bin/bash
#
# Wrapper script for Nextflow
# In Docker: launches from NXF_LAUNCH_DIR (work dir) so .nextflow/ persists for resume.
# Locally:   launches from the project directory.
#

# Guard: gosu with an unknown UID may reset HOME to "/" (UID not in /etc/passwd).
# Ensure HOME is a writable path before anything else.
if [ -z "$HOME" ] || [ "$HOME" = "/" ]; then
    export HOME=/tmp/home
    mkdir -p "$HOME" 2>/dev/null || true
fi

# Set Nextflow home directory (for global cache, history, etc.)
export NXF_HOME="${NXF_HOME:-$HOME/.nextflow}"

# Pin the Nextflow version for local runs, matching the Docker image.
# Without this, a freshly installed launcher self-downloads the newest release;
# Nextflow 26.x defaults to the strict config parser, which rejects the Groovy in
# nextflow.config (nvidia-smi .execute(), Math.floor, try/catch) and fails at parse time.
# In Docker, NXF_VER is already exported by the image, so the :- default is a no-op there.
# Keep in sync with ARG NEXTFLOW_VERSION in the Dockerfile.
export NXF_VER="${NXF_VER:-25.10.2}"

# Log path must live under NXF_HOME when NXF_LOG is unset. Do not use $HOME here:
# after gosu to a numeric UID, HOME may be "/" so $HOME/.nextflow becomes //.nextflow
# and mkdir fails with permission denied.
export NXF_LOG="${NXF_LOG:-$NXF_HOME/logs/nextflow.log}"

# Create directories if they don't exist
mkdir -p "$(dirname "$NXF_LOG")"
mkdir -p "$NXF_HOME"

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared CLI-flag helpers: KNOWN_FLAGS (from known_flags.txt), normalize_flag,
# validate_flags, print_usage. Single source of truth shared with entrypoint.sh + main.nf.
# Guarded so a missing flags.sh degrades gracefully (fail-open) instead of breaking runs.
# shellcheck source=flags.sh
if [ -f "$SCRIPT_DIR/flags.sh" ]; then
    . "$SCRIPT_DIR/flags.sh"
else
    echo "WARNING: flags.sh not found; --help and unknown-argument validation are disabled." >&2
    validate_flags() { UNKNOWN_FLAGS=(); return 0; }
    normalize_flag() { printf '%s' "$1"; }
    print_usage() { echo "brainana — see docs/usage_notes.rst"; }
fi

# Find Nextflow executable
if [ -f "$SCRIPT_DIR/nextflow" ] && [ -x "$SCRIPT_DIR/nextflow" ]; then
    NEXTFLOW="$SCRIPT_DIR/nextflow"
elif command -v nextflow &> /dev/null; then
    NEXTFLOW="nextflow"
else
    echo "Error: Nextflow not found" >&2
    exit 1
fi

# Launch directory — where Nextflow creates .nextflow/ (history + cache for resume).
#   Docker: NXF_LAUNCH_DIR is set by entrypoint.sh (persistent work dir).
#   Local:  Derived from --work_dir or --work-dir arg if present, else SCRIPT_DIR.
if [ -z "$NXF_LAUNCH_DIR" ]; then
    _prev=""
    for _arg in "$@"; do
        # Space form: --work_dir /path  or  --work-dir /path
        if [ "$_prev" = "--work_dir" ] || [ "$_prev" = "--work-dir" ]; then
            NXF_LAUNCH_DIR="$_arg"
            break
        fi
        # Equals form: --work_dir=/path  or  --work-dir=/path
        case "$_arg" in
            --work_dir=*|--work-dir=*) NXF_LAUNCH_DIR="${_arg#*=}"; break ;;
        esac
        _prev="$_arg"
    done
fi
if [ -n "$NXF_LAUNCH_DIR" ]; then
    mkdir -p "$NXF_LAUNCH_DIR" 2>/dev/null || true
fi
cd "${NXF_LAUNCH_DIR:-$SCRIPT_DIR}"

# Build the command
# Always use -log to redirect log files and -C to specify config
CMD_ARGS=(-log "$NXF_LOG" -C "$SCRIPT_DIR/nextflow.config")

# Function to extract parameter value from arguments
# Handles both --param=value and --param value formats
extract_param() {
    local param_name="$1"
    local args=("${@:2}")
    local i=0
    while [ $i -lt ${#args[@]} ]; do
        local arg="${args[$i]}"
        # Check for --param=value format
        if [[ "$arg" == --${param_name}=* ]]; then
            echo "${arg#*=}"
            return 0
        fi
        # Check for --param value format
        if [[ "$arg" == --${param_name} ]]; then
            # Check if next argument exists (not empty and not another flag)
            if [ $((i + 1)) -lt ${#args[@]} ] && [[ "${args[$((i + 1))]}" != -* ]]; then
                echo "${args[$((i + 1))]}"
                return 0
            fi
        fi
        ((i++))
    done
    return 1  # Not found
}

# CLI flag connector normalization + the KNOWN_FLAGS allowlist now live in flags.sh
# (sourced above), shared with entrypoint.sh and mirrored by main.nf's guard.

# Default config when --config_file / --config not provided (same as Nextflow default)
DEFAULT_CONFIG="$SCRIPT_DIR/src/nhp_mri_prep/config/defaults.yaml"

# Function to run BIDS discovery before Nextflow
run_bids_discovery() {
    local args=("$@")
    
    # Extract required parameters (config_file has default)
    local bids_dir=$(extract_param "bids_dir" "${args[@]}")
    local output_dir=$(extract_param "output_dir" "${args[@]}")
    local config_file=$(extract_param "config_file" "${args[@]}")
    [ -z "$config_file" ] && config_file="$DEFAULT_CONFIG"
    
    # Check if required path parameters are present
    if [ -z "$bids_dir" ] || [ -z "$output_dir" ]; then
        return 0  # Skip discovery if params not available (Nextflow will error later)
    fi
    
    # Extract optional parameters
    local subjects=$(extract_param "subjects" "${args[@]}")
    local sessions=$(extract_param "sessions" "${args[@]}")
    local tasks=$(extract_param "tasks" "${args[@]}")
    local runs=$(extract_param "runs" "${args[@]}")

    # --anat_only is a boolean: a bare flag means true, an explicit value is
    # honoured (Nextflow reads `--anat_only false` as false). Forwarded so the
    # discovery summary and job lists match what main.nf will actually run.
    local anat_only=""
    local a
    for a in "${args[@]}"; do
        [[ "$a" == "--anat_only" || "$a" == --anat_only=* ]] && anat_only="true"
    done
    if [ -n "$anat_only" ]; then
        local anat_only_value=$(extract_param "anat_only" "${args[@]}")
        case "${anat_only_value,,}" in
            false|0|no) anat_only="" ;;
        esac
    fi
    
    # Build discovery command
    local discovery_script="$SCRIPT_DIR/src/nhp_mri_prep/nextflow_scripts/discover_bids_for_nextflow.py"
    
    if [ ! -f "$discovery_script" ]; then
        echo "Warning: Discovery script not found: $discovery_script" >&2
        return 1
    fi
    
    echo "============================================"
    echo "Running BIDS Discovery"
    echo "============================================"
    
    # Build command
    local cmd=("python3" "$discovery_script")
    cmd+=("--bids_dir" "$bids_dir")
    cmd+=("--output_dir" "$output_dir")
    cmd+=("--config_file" "$config_file")
    
    [ -n "$subjects" ] && cmd+=("--subjects" "$subjects")
    [ -n "$sessions" ] && cmd+=("--sessions" "$sessions")
    [ -n "$tasks" ] && cmd+=("--tasks" "$tasks")
    [ -n "$runs" ] && cmd+=("--runs" "$runs")
    [ -n "$anat_only" ] && cmd+=("--anat_only")
    
    # Run discovery
    "${cmd[@]}"
    local discovery_exit=$?
    if [ $discovery_exit -ne 0 ]; then
        echo "ERROR: Aborting pipeline. Fix the issues above and re-run." >&2
        exit $discovery_exit
    fi
    
    echo "============================================"
    echo ""
}

# Show brainana usage for -h/--help anywhere in the args, before dispatch/discovery/Nextflow.
for _a in "$@"; do
    case "$_a" in
        -h|--help) print_usage; exit 0 ;;
    esac
done

# Handle different Nextflow commands
if [ $# -eq 0 ]; then
    # No arguments - show help
    exec "$NEXTFLOW" "${CMD_ARGS[@]}" "$@"
elif [ "$1" = "run" ]; then
    # Determine workflow file and remaining arguments
    workflow_file=""
    remaining_args=()
    
    if [ $# -gt 1 ] && [ -f "$SCRIPT_DIR/$2" ]; then
        # Workflow file found relative to project dir
        workflow_file="$SCRIPT_DIR/$2"
        remaining_args=("${@:3}")
    elif [ $# -gt 1 ] && [ -f "$2" ]; then
        # Absolute path to workflow file
        workflow_file="$2"
        remaining_args=("${@:3}")
    elif [ $# -gt 1 ] && [[ "$2" == *.nf ]]; then
        # User specified a .nf file name — resolve relative to project
        workflow_file="$SCRIPT_DIR/$2"
        remaining_args=("${@:3}")
    else
        # No workflow file specified, use project's main.nf
        workflow_file="$SCRIPT_DIR/main.nf"
        remaining_args=("${@:2}")
    fi
    
    # Normalize CLI flag connectors (hyphen -> underscore for known flags) so all
    # downstream handling and Nextflow see canonical snake_case forms. This is the
    # single choke point for both the Docker and local entry paths.
    normalized_conn=()
    for arg in "${remaining_args[@]}"; do
        normalized_conn+=("$(normalize_flag "$arg")")
    done
    remaining_args=("${normalized_conn[@]}")

    # Consume --freesurfer_license: it only configures the FS_LICENSE env var (used by
    # FreeSurfer/surface reconstruction); it is NOT a Nextflow param, so strip it before
    # forwarding. This avoids an unused param, keeps the unknown-flag guard clean, and
    # lets local runs set FS_LICENSE from the flag. In Docker, entrypoint.sh already
    # exported FS_LICENSE and no longer forwards the flag.
    fs_lic=$(extract_param "freesurfer_license" "${remaining_args[@]}")
    [ -n "$fs_lic" ] && export FS_LICENSE="${FS_LICENSE:-$fs_lic}"
    stripped_args=()
    j=0
    while [ $j -lt ${#remaining_args[@]} ]; do
        a="${remaining_args[$j]}"
        if [[ "$a" == --freesurfer_license=* ]]; then
            ((j++)); continue
        fi
        if [[ "$a" == --freesurfer_license ]]; then
            ((j++)); [ $j -lt ${#remaining_args[@]} ] && ((j++)); continue
        fi
        stripped_args+=("$a")
        ((j++))
    done
    remaining_args=("${stripped_args[@]}")

    # Filter out --no_docker flag and set environment variable if present
    filtered_args=()
    for arg in "${remaining_args[@]}"; do
        if [[ "$arg" == --no_docker ]]; then
            # Set environment variable to disable Docker
            export NXF_NO_DOCKER=1
            # Skip this argument
        else
            # Keep this argument
            filtered_args+=("$arg")
        fi
    done
    remaining_args=("${filtered_args[@]}")
    
    # For main.nf: normalize config (--config and --config_file are aliases; default to DEFAULT_CONFIG).
    # (Connector normalization already ran above, so work-dir is canonical --work_dir here.)
    if [[ "$workflow_file" == "main.nf" ]] || [[ "$workflow_file" == */main.nf ]]; then
        # Fail fast on unrecognized arguments, BEFORE BIDS discovery / Nextflow, so a typo
        # like --output_space__ errors in ~1s instead of deep inside the pipeline (which
        # would also leave a partial QC report). main.nf re-checks as a backstop.
        if ! validate_flags "${remaining_args[@]}"; then
            echo "Unknown argument(s): ${UNKNOWN_FLAGS[*]}" >&2
            echo "" >&2
            print_usage >&2
            echo "" >&2
            echo "Hint: custom templates use --output_space <file>, not --custom-template." >&2
            exit 2
        fi

        effective_config=$(extract_param "config_file" "${remaining_args[@]}")
        [ -z "$effective_config" ] && effective_config=$(extract_param "config" "${remaining_args[@]}")
        [ -z "$effective_config" ] && effective_config="$DEFAULT_CONFIG"
        
        # Build normalized args: drop --config/--config_file and their values; append the resolved --config_file
        normalized_args=()
        i=0
        while [ $i -lt ${#remaining_args[@]} ]; do
            arg="${remaining_args[$i]}"
            if [[ "$arg" == --config=* ]]; then
                ((i++))
                continue
            fi
            if [[ "$arg" == --config ]]; then
                ((i++))
                [ $i -lt ${#remaining_args[@]} ] && ((i++))
                continue
            fi
            if [[ "$arg" == --config_file=* ]]; then
                ((i++))
                continue
            fi
            if [[ "$arg" == --config_file ]]; then
                ((i++))
                [ $i -lt ${#remaining_args[@]} ] && ((i++))
                continue
            fi
            normalized_args+=("$arg")
            ((i++))
        done
        normalized_args+=("--config_file" "$effective_config")
        
        # Run BIDS discovery with normalized args (always has --config_file).
        # Skipped in dry-run mode, which only echoes the assembled Nextflow command.
        [ "${BRAINANA_DRY_RUN:-0}" = "1" ] || run_bids_discovery "${normalized_args[@]}"
    fi
    
    # Use normalized_args for Nextflow if we built them (main.nf), else remaining_args
    if [ -n "${normalized_args+x}" ]; then
        final_args=("${normalized_args[@]}")
    else
        final_args=("${remaining_args[@]}")
    fi
    # Dry-run: print the exact Nextflow invocation and exit without running it, so
    # different flag spellings can be checked to resolve to an identical command.
    if [ "${BRAINANA_DRY_RUN:-0}" = "1" ]; then
        printf '%s\n' "$NEXTFLOW" "${CMD_ARGS[@]}" run "$workflow_file" "${final_args[@]}"
        exit 0
    fi
    exec "$NEXTFLOW" "${CMD_ARGS[@]}" run "$workflow_file" "${final_args[@]}"
else
    # Other Nextflow commands (info, clean, etc.) - pass through as-is
    # But still run from RUN_DIR to keep project clean
    exec "$NEXTFLOW" "${CMD_ARGS[@]}" "$@"
fi


