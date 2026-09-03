/*
 * Anatomical processing modules for nhp_mri_prep Nextflow pipeline
 */

process ANAT_SYNTHESIS {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${(session_id && session_id != '') ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*.{nii.gz,json}',
        saveAs: { filename -> filename == 'metadata.json' ? null : filename }

    input:
    tuple val(subject_id), val(session_id), path(anat_files)
    path config_file
    
    output:
    tuple val(subject_id), val(session_id), path("*.nii.gz"), file("bids_name.txt"), emit: synthesized
    // Must be "*.json", not "metadata.json": publishDir only ever publishes files that
    // are *declared outputs*, so narrowing this to metadata.json meant the derivative
    // sidecar was written into the task dir and then silently dropped -- the publishDir
    // pattern below says `json` and the saveAs closure exists precisely to suppress
    // metadata.json, but there was no other json output for it to let through. Every
    // other process in this file already declares "*.json" for the same reason.
    path "*.json", emit: metadata

    script:
    def anat_files_list = anat_files.collect { "'${it}'" }.join(', ')
    def first_file = anat_files[0]
    """
    \${PYTHON:-python3} <<'PYTHON_EOF' > /dev/null
from pathlib import Path

from nhp_mri_prep.steps.anatomical import anat_synthesis
from nhp_mri_prep.utils.bids import create_synthesized_bids_filename
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, normalize_session_id, save_metadata, create_output_link
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar

# Load config
config = load_config('${config_file}')

# Empty session_id on channel => subject-level synthesis job
session_id_raw = '${session_id}'
session_id = normalize_session_id(session_id_raw)
is_subject_level = session_id is None

# Get anatomical files
anat_files = [Path(f) for f in [${anat_files_list}]]

# BIDS naming template (first input)
bids_name = Path('${first_file}')

# Run synthesis (anat_synthesis function works for all anatomical modalities via underlying synthesis_multiple_anatomical)
result = anat_synthesis(
    anat_files=anat_files,
    working_dir=Path('work'),
    config=config
)

synthesized = result.metadata.get("synthesized", False)

# Ingest normalization findings. Every key below is omitted entirely when nothing
# fired, so a well-formed dataset gets byte-identical sidecars. Container format
# (.nii -> .nii.gz) is deliberately not recorded: it changes nothing about the data.
reports = result.metadata.get("input_normalization", [])

# 4D->3D collapse: a trailing singleton frame axis dropped, or a genuine
# multi-volume anatomical averaged over its last axis.
dim_actions = [r["dim"] for r in reports if r.get("dim", "unchanged") != "unchanged"]

# Orientation recovered for inputs that stored none (qform_code == 0 and
# sform_code == 0). Its presence means the geometry of this image is an assumed
# convention, not something the acquisition recorded -- see the left/right caveat.
orientation_actions = [
    r["orientation"] for r in reports if r.get("orientation", "unchanged") != "unchanged"
]

# qform and sform disagreed and were resolved toward the sform, so ANTs and
# FastSurfer now read the same grid that registration uses.
geometry_actions = [
    r["geometry"] for r in reports if r.get("geometry", "unchanged") != "unchanged"
]

# Header defects with no safe automatic repair (non-mm units, pixdim/affine
# mismatch). Reported rather than fixed -- these also surface in the QC report.
header_warnings = [w for r in reports for w in r.get("warnings", [])]

modality = detect_modality(bids_name)
bids_output_filename, bids_name_for_downstream = create_synthesized_bids_filename(
    original_file=bids_name,
    modality=modality,
    is_subject_level=is_subject_level,
    synthesized=synthesized,
)

create_output_link(result.output_file, bids_output_filename)

# JSON sidecar (synthesized anatomical, native scanner space -> no template block)
sidecar_extra = {"Synthesized": synthesized}
if dim_actions:
    sidecar_extra["Input4DCollapsed"] = dim_actions
if orientation_actions:
    sidecar_extra["OrientationRecovered"] = orientation_actions
if geometry_actions:
    sidecar_extra["QformSformReconciled"] = geometry_actions
if header_warnings:
    sidecar_extra["InputHeaderWarnings"] = header_warnings
write_derivative_sidecar(
    bids_output_filename,
    skull_stripped=False,
    sources=[str(f) for f in anat_files],
    extra=sidecar_extra,
)

# Save metadata
save_metadata(result.metadata)

# Write the appropriate path to file for Nextflow value output
with open('bids_name.txt', 'w') as f:
    f.write(bids_name_for_downstream)
PYTHON_EOF
    """
}

process ANAT_CONFORM {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*.{mat,nii.gz,json}',
        saveAs: { filename ->
            // Exclude the QC-only references (template_resampled.nii.gz, its full-FOV
            // counterpart and the FOV box mask), the _desc-conform_ intermediate (not for
            // publication) and the generic per-step metadata.json (sidecars are published,
            // metadata.json is not). The desc token is matched exactly: _desc-conformFullFOV_
            // IS published, and a substring test would silently swallow it.
            if (filename in ['template_resampled.nii.gz', 'template_for_xfm_fullfov.nii.gz', 'conform_fov_box.nii.gz', 'metadata.json'] || filename.contains('_desc-conform_')) {
                return null
            }
            return filename
        }
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name)
    path config_file
    
    output:
    // Exact token: must not also capture *_desc-conformFullFOV_*, or downstream steps
    // get a two-element list where they expect one file.
    tuple val(subject_id), val(session_id), path("*_desc-conform_*.nii.gz"), val(bids_name), emit: output
    // Transforms: [sub, ses, forward_transform, inverse_transform]
    tuple val(subject_id), val(session_id), path("*from-scanner_to-*_mode-image_xfm.{h5,mat,nii.gz}"), path("*from-*_to-scanner_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    // Reference: [sub, ses, reference]
    tuple val(subject_id), val(session_id), path("template_resampled.nii.gz"), emit: reference
    // Full FOV: [sub, ses, uncropped_image, uncropped_template, fov_box] - QC only.
    // optional: true because this is a leaf diagnostic. _conform_full_fov degrades to the
    // target FOV on its own two inner failures, but a raise outside them (the nib.load
    // header reads, shutil.copy2 on the no-expansion path, write_inner_box_mask) would
    // otherwise leave these three declared-but-absent and fail ANAT_CONFORM -- and with it
    // the whole anatomical workflow, since this process has no errorStrategy. All three
    // stand or fall together: a partial trio would mix grids in the QC figure.
    tuple val(subject_id), val(session_id), path("*_desc-conformFullFOV_*.nii.gz"), path("template_for_xfm_fullfov.nii.gz"), path("conform_fov_box.nii.gz"), emit: full_fov, optional: true
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_conform
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.templates import resolve_template, space_label_for
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, save_metadata, create_output_link
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path
import shutil
import os

# Load config
config = load_config('${config_file}')

# Get BIDS naming template (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Determine modality from BIDS naming template filename
modality = detect_modality(bids_name)

# Get effective_output_space from effective config file
config = load_config('${config_file}')
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')

# Resolve template
template_file = Path(resolve_template(effective_output_space))

# Create step input
input_obj = StepInput(
    input_file=Path('${input_file}'),
    working_dir=Path('work'),
    config=config,
    output_name='anat_conformed.nii.gz',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}'
    }
)

# Run step
result = anat_conform(input_obj, template_file=template_file)

# Generate BIDS-compliant output filename
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-conform',
    modality=modality
)

# Use symlink to avoid duplication - Nextflow publishDir will handle final copy
create_output_link(result.output_file, bids_output_filename)

# Full-FOV conform: the same conform on a grid enlarged so no scanner-space voxel is
# cropped (recording chamber, head-post, neck). Leaf output - nothing downstream reads it.
if "conformed_full_fov" in result.additional_files:
    # space-{modality}: this image is on the conform grid, i.e. the same space as
    # space-T1w_desc-preproc_T1w (identical direction/spacing, an integer voxel offset
    # away) — NOT the scanner space of the input. Passing the space in the suffix also
    # makes create_bids_output_filename drop the input's own space-scanner entity.
    full_fov_filename = create_bids_output_filename(
        original_file_path=bids_name,
        suffix=f'space-{modality}_desc-conformFullFOV',
        modality=modality
    )
    create_output_link(result.additional_files["conformed_full_fov"], full_fov_filename)
    write_derivative_sidecar(
        full_fov_filename,
        sources=[str(Path('${input_file}'))],
        extra={'FullFOVPadding': result.metadata.get('full_fov_pad')},
    )

# QC-only references for the conform snapshot (staged at root, never published)
for _key, _link_name in [
    ("template_resampled_full_fov", 'template_for_xfm_fullfov.nii.gz'),
    ("fov_box", 'conform_fov_box.nii.gz'),
]:
    _src = result.additional_files.get(_key)
    if _src is not None and Path(_src).exists():
        create_output_link(_src, _link_name)

# Generate BIDS prefix (filename stem without modality)
original_stem = get_filename_stem(bids_name)
bids_prefix = original_stem.replace(f"_{modality}", "")

# Copy transform files with BIDS-compliant names
if "forward_transform" in result.additional_files:
    # Forward transform: from-scanner_to-{modality}
    bids_transform_name = f"{bids_prefix}_from-scanner_to-{modality}_mode-image_xfm.mat"
    shutil.copy2(result.additional_files["forward_transform"], bids_transform_name)
    # Sidecar (scanner<->native transform; no template block)
    write_derivative_sidecar(bids_transform_name, engine=result.metadata.get('engine'), sources=[str(Path('${input_file}'))])

if "inverse_transform" in result.additional_files:
    # Inverse transform: from-{modality}_to-scanner
    bids_transform_name = f"{bids_prefix}_from-{modality}_to-scanner_mode-image_xfm.mat"
    shutil.copy2(result.additional_files["inverse_transform"], bids_transform_name)
    write_derivative_sidecar(bids_transform_name, engine=result.metadata.get('engine'), sources=[str(Path('${input_file}'))])

# Create symlink to reference file for QC (file is in work/ subdirectory, needs to be at root for Nextflow output)
# Use create_output_link() for consistency and proper symlink resolution
if "template_resampled" in result.additional_files:
    reference_src = result.additional_files["template_resampled"]
    if reference_src.exists():
        # Create symlink at root level so Nextflow output pattern can find it
        reference_dest = Path('template_resampled.nii.gz')
        create_output_link(reference_src, str(reference_dest))

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process ANAT_BIAS_CORRECTION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name)
    tuple val(subject_id), val(session_id), path(mask_file)
    path config_file
    
    output:
    // Bias-corrected full head output: [sub, ses, biascorrected_file, bids_template]
    // Use T?w pattern to match both T1w and T2w modalities
    tuple val(subject_id), val(session_id), path("*desc-biascorrect*_T?w.nii.gz"), val(bids_name), emit: output
    // Brain output: [sub, ses, brain_file] - will be joined with bids_template in workflow
    // Use desc-biascorrect naming (not desc-preproc) - publishing step will handle preproc naming
    // Use T?w pattern to match both T1w and T2w modalities
    tuple val(subject_id), val(session_id), path("*desc-biascorrect*_T?w_brain.nii.gz"), emit: brain
    path "*.json", emit: metadata
    
    script:
    """
    # Thread environment variables are set by Nextflow's beforeScript based on task.cpus
    # Python code reads OMP_NUM_THREADS from environment
    
    \${PYTHON:-python3} <<'PYTHON_EOF'
from nhp_mri_prep.steps.anatomical import anat_bias_correction
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import create_bids_output_filename
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, save_metadata, create_output_link
)
from pathlib import Path
import shutil
import os
import json

# Load config
config = load_config('${config_file}')

# Get BIDS naming template (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Determine modality from BIDS naming template filename
modality = detect_modality(bids_name)

# Get mask file (already validated as real in workflow)
mask_file = Path('${mask_file}')
brain_mask = mask_file

# Create step input
input_obj = StepInput(
    input_file=Path('${input_file}'),
    working_dir=Path('work'),
    config=config,
    output_name='anat_bias_corrected.nii.gz',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}'
    }
)

# Run step
result = anat_bias_correction(input_obj, brain_mask=brain_mask)

# Generate BIDS-compliant output filename (for internal workflow use)
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-biascorrect',
    modality=modality
)

# Use symlink to avoid duplication - publishing step will handle final copy
create_output_link(result.output_file, bids_output_filename)

# Always output brain (real if available, else same file as output) for consistent structure
# Structure: [sub, ses, brain_file] - will be joined with bids_template in workflow
# Use desc-biascorrect naming (not desc-preproc) - publishing step will handle preproc naming
from nhp_mri_prep.utils.bids import get_filename_stem
original_stem = get_filename_stem(bids_name)
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")
bids_brain_filename = f"{bids_prefix_wo_modality}_desc-biascorrect_{modality}_brain.nii.gz"

if "brain" in result.additional_files:
    # Real brain was generated - use it
    create_output_link(result.additional_files["brain"], bids_brain_filename)
else:
    # No brain from step (e.g. no mask) - use bias-corrected full head as brain for downstream (e.g. registration)
    create_output_link(result.output_file, bids_brain_filename)

# Save metadata
with open('metadata.json', 'w') as f:
    json.dump(result.metadata, f, indent=2)
PYTHON_EOF
    """
}

process ANAT_SKULLSTRIPPING {
    label 'gpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*desc-brain*',
        enabled: true
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name)
    path config_file
    val gpu_id
    
    output:
    // Pattern matches files with desc-skullstrip (full head, inherited from input)
    tuple val(subject_id), val(session_id), path("*desc-skullstrip*_T1w.nii.gz"), val(bids_name), emit: output  // Full head (inherited from input, not skullstripped)
    tuple val(subject_id), val(session_id), path("*desc-skullstrip*_T1w_brain.nii.gz"), val(bids_name), emit: brain  // Brain-only (skullstripped, generated by this step)
    // Note: For T2w processing, a separate ANAT_SKULLSTRIPPING_T2W process would be used with T2w-specific patterns
    tuple val(subject_id), val(session_id), path("*_desc-brain_mask.nii.gz"), emit: brain_mask
    tuple val(subject_id), val(session_id), path("*_desc-brain_hemimask.nii.gz"), optional: true, emit: brain_hemimask
    tuple val(subject_id), val(session_id), path("*_desc-brain_atlas*.nii.gz"), optional: true, emit: brain_segmentation
    tuple val(subject_id), val(session_id), path("*_desc-brain_atlas*.tsv"), optional: true, emit: brain_segmentation_lut
    val gpu_id, emit: gpu_token
    path "*.json", emit: metadata
    
    script:
    """
    # GPU Assignment (gpu_id is 'none' when workflow GPU scheduling is disabled, e.g. CPU mode)
    if [ "${gpu_id}" != "none" ]; then
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        echo "[GPU Assignment] Task ${task.index} -> GPU ${gpu_id} (of ${params.gpu_count} available)"
    else
        export CUDA_VISIBLE_DEVICES=""
    fi

    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_skullstripping
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import create_bids_output_filename
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, save_metadata, create_output_link
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path
import shutil
import os

# Load config
config = load_config('${config_file}')

# Get BIDS naming template (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Determine modality from BIDS naming template filename
modality = detect_modality(bids_name)

# Create step input
input_obj = StepInput(
    input_file=Path('${input_file}'),
    working_dir=Path('work'),
    config=config,
    output_name='anat_brain.nii.gz',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}'
    }
)

# Run step
result = anat_skullstripping(input_obj)

# Generate BIDS-compliant output filenames
# Principle: anat_after_xxxstep = full head (_T1w), anat_after_xxxstep_brain = brain (_T1w_brain)
# Use desc-skullstrip naming to indicate this step (similar to desc-conform, desc-biascorrect)
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
original_stem = get_filename_stem(bids_name)
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")

# Output 1: Full head version (not skullstripped) - inherit from input file
# ANAT_SKULLSTRIPPING does NOT generate a full head file - it only generates the brain version
# So we inherit the input file (full head) and create a symlink with proper BIDS naming
# Format: {prefix}_desc-skullstrip_{modality}.nii.gz (e.g., sub-XXX_ses-001_run-1_desc-skullstrip_T1w.nii.gz)
bids_output_full_head = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-skullstrip',
    modality=modality
)
create_output_link(Path('${input_file}'), bids_output_full_head)

# Output 2: Brain-only version (skullstripped) - generated by this step
# Format: {prefix}_desc-skullstrip_{modality}_brain.nii.gz (e.g., sub-XXX_ses-001_run-1_desc-skullstrip_T1w_brain.nii.gz)
# Note: _brain is appended after the modality, not part of it
bids_output_brain = f"{bids_prefix_wo_modality}_desc-skullstrip_{modality}_brain.nii.gz"
create_output_link(result.output_file, bids_output_brain)

# Create symlinks for additional files with BIDS-compliant names
# Keep large files (masks, segmentations) as symlinks until published - saves storage
atlas_name = result.metadata.get('atlas_name')

if "brain_mask" in result.additional_files:
    bids_additional_name = f"{bids_prefix_wo_modality}_space-T1w_desc-brain_mask.nii.gz"
    create_output_link(result.additional_files["brain_mask"], bids_additional_name)
    # Sidecar (brain mask, native T1w space -> no template block)
    write_derivative_sidecar(bids_additional_name, roi_type="Brain", sources=[str(Path('${input_file}'))])

if "segmentation" in result.additional_files:
    if atlas_name:
        bids_additional_name = f"{bids_prefix_wo_modality}_space-T1w_desc-brain_atlas{atlas_name}.nii.gz"
    else:
        bids_additional_name = f"{bids_prefix_wo_modality}_space-T1w_desc-brain_segmentation.nii.gz"
    create_output_link(result.additional_files["segmentation"], bids_additional_name)
    write_derivative_sidecar(
        bids_additional_name,
        sources=[str(Path('${input_file}'))],
        extra={"Atlas": atlas_name} if atlas_name else None,
    )

if "hemimask" in result.additional_files:
    bids_additional_name = f"{bids_prefix_wo_modality}_space-T1w_desc-brain_hemimask.nii.gz"
    create_output_link(result.additional_files["hemimask"], bids_additional_name)
    write_derivative_sidecar(bids_additional_name, roi_type="ROI", sources=[str(Path('${input_file}'))])

if "input_cropped" in result.additional_files:
    # Keep original name for input_cropped (or define BIDS name if needed)
    create_output_link(result.additional_files["input_cropped"], result.additional_files["input_cropped"].name)

if "atlas_lut" in result.additional_files:
    # LUT same base as segmentation: desc-brain_atlas{atlas_name}.tsv
    if atlas_name:
        bids_additional_name = f"{bids_prefix_wo_modality}_space-T1w_desc-brain_atlas{atlas_name}.tsv"
    else:
        bids_additional_name = f"{bids_prefix_wo_modality}_space-T1w_desc-brain_segmentation.tsv"
    create_output_link(result.additional_files["atlas_lut"], bids_additional_name)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process ANAT_SURFACE_RECONSTRUCTION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    // Allow workflow to continue if surface reconstruction fails for some subjects
    // (e.g., due to poor image quality). Failed subjects won't emit outputs,
    // so downstream QC processes will simply skip them.
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/fastsurfer",
        mode: 'copy',
        pattern: 'fastsurfer/**',
        saveAs: { filename -> 
            // Strip 'fastsurfer/' prefix to move sub-XXX directly to output_dir/fastsurfer/sub-XXX
            filename.replace('fastsurfer/', '')
        }
    
    input:
    tuple val(subject_id), val(session_id), path(t1w_file), val(bids_name), path(segmentation_file), path(brain_mask), path(arm6_atlas_file), val(session_count)
    path config_file
    
    output:
    // Output path: sub-XXX or sub-XXX_ses-XXX (determined by script based on session_count)
    // Script creates the directory with the correct name that matches what Nextflow expects
    // When session_count > 1: creates fastsurfer/sub-XXX_ses-XXX
    // When session_count == 1: creates fastsurfer/sub-XXX
    // Use a glob pattern to match the directory name
    tuple val(subject_id), val(session_id), path("fastsurfer/sub-${subject_id}*"), emit: subject_dir
    tuple val(subject_id), val(session_id), path("actual_subject_id.txt"), emit: actual_subject_id
    tuple val(subject_id), val(session_id), path("metadata.json"), emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_surface_reconstruction
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.nextflow import (
    load_config, normalize_session_id, save_metadata
)
from pathlib import Path
import os

# Load config
config = load_config('${config_file}')

# Handle empty string session_id (subject-level synthesis)
session_id_raw = '${session_id}'
session_id = normalize_session_id(session_id_raw)
session_count = int('${session_count}') if '${session_count}' else 1

# Create step input
input_obj = StepInput(
    input_file=Path('${t1w_file}'),
    working_dir=Path('work'),
    config=config,
    output_name='surface_reconstruction',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': session_id,
        'session_count': session_count
    }
)

# Get brain mask if provided (check if file exists and is not empty)
brain_mask_path = None
if '${brain_mask}' and Path('${brain_mask}').exists() and Path('${brain_mask}').stat().st_size > 0:
    brain_mask_path = Path('${brain_mask}')

# Get ARM6 atlas if provided (check if file exists and is not empty)
arm6_atlas_path = None
if '${arm6_atlas_file}' and Path('${arm6_atlas_file}').exists() and Path('${arm6_atlas_file}').stat().st_size > 0:
    arm6_atlas_path = Path('${arm6_atlas_file}')

# Run surface reconstruction
result = anat_surface_reconstruction(
    input_obj,
    t1w_file=Path('${t1w_file}'),
    segmentation_file=Path('${segmentation_file}'),
    brain_mask=brain_mask_path,
    arm6_atlas=arm6_atlas_path
)

# Copy directory to work directory root so Nextflow can find it
# The output is created in work/fastsurfer/sub-XXX or work/fastsurfer/sub-XXX_ses-XXX
# Nextflow expects fastsurfer/sub-${subject_id} (which may or may not include session)
# We need to ensure the path matches what Nextflow expects
output_subject_dir = result.output_file
# The result.output_file already has the correct subject ID (with or without session)
# Extract just the directory name (e.g., 'sub-XXX' or 'sub-XXX_ses-XXX')
actual_subject_id = output_subject_dir.name

# Determine what Nextflow expects based on session_count
# If session_count > 1 and session_id exists, Nextflow should expect sub-XXX_ses-XXX
# Otherwise, it expects sub-XXX
# session_id and session_count already defined above

if session_id and session_count > 1:
    # Multiple sessions: Nextflow expects sub-XXX_ses-XXX
    ses_id = session_id if not session_id.startswith("ses-") else session_id[4:]
    base_subject_id = 'sub-${subject_id}' if '${subject_id}'.startswith('sub-') else f"sub-${subject_id}"
    expected_subject_id = f"{base_subject_id}_ses-{ses_id}"
else:
    # Single session or no session: Nextflow expects sub-XXX
    expected_subject_id = 'sub-${subject_id}' if '${subject_id}'.startswith('sub-') else f"sub-${subject_id}"

expected_path = Path('fastsurfer') / expected_subject_id

# Ensure the output directory exists
if not output_subject_dir.exists():
    raise FileNotFoundError(f"Surface reconstruction output not found: {output_subject_dir}")

# Get absolute paths for reliable copying
output_abs = output_subject_dir.resolve()
expected_abs = expected_path.resolve()

# Create the expected path location
expected_abs.parent.mkdir(parents=True, exist_ok=True)

# Remove existing symlink/directory if it exists
if expected_abs.exists() or expected_abs.is_symlink():
    import shutil
    if expected_abs.is_symlink() or expected_abs.is_file():
        expected_abs.unlink()
    elif expected_abs.is_dir():
        shutil.rmtree(expected_abs)

# Always copy the directory (not symlink) to ensure actual content is moved
import shutil
shutil.copytree(output_abs, expected_abs, dirs_exist_ok=True)

# Verify the expected path exists (check relative path for Nextflow)
if not expected_path.exists():
    raise FileNotFoundError(f"Failed to create expected output path: {expected_path} (absolute: {expected_abs})")

# Write actual subject ID to a file for downstream processes
with open('actual_subject_id.txt', 'w') as f:
    f.write(actual_subject_id)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process ANAT_REGISTRATION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*.{nii.gz,h5,mat,json}',
        saveAs: { filename -> (filename.contains('ref_from_anat_reg.nii.gz') || filename == 'metadata.json') ? null : filename }
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name), path(unskullstripped_file)
    path config_file  // Effective config file with all resolved parameters
    val gpu_id  // GPU ID for scheduling ('none' for CPU mode, integer for GPU mode)
    
    output:
    // Output: [sub, ses, registered_file, bids_template]
    tuple val(subject_id), val(session_id), path("*space-*.nii.gz"), val(bids_name), emit: output
    // Transforms: [sub, ses, bids_name, forward_transform, inverse_transform]
    // Use patterns that match the actual naming: forward starts with from-T1w_to- or from-T2w_to-, inverse ends with _to-T1w or _to-T2w
    tuple val(subject_id), val(session_id), val(bids_name), path("*from-T1w_to-*_mode-image_xfm.{h5,mat,nii.gz}"), path("*from-*_to-T1w_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    // Reference: [sub, ses, reference_file]
    tuple val(subject_id), val(session_id), path("*ref_from_anat_reg.nii.gz"), emit: reference
    path "*.json", emit: metadata
    val gpu_id, emit: gpu_token
    
    script:
    """
    # Conditional GPU assignment
    if [ "${gpu_id}" != "none" ]; then
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        echo "[GPU Assignment] Task ${task.index} -> GPU ${gpu_id} (of ${params.gpu_count} available)"
    fi
    
    # Get effective_output_space from effective config file
    EFFECTIVE_OUTPUT_SPACE=\$(\${PYTHON:-python3} <<'PYTHON_OUTPUT_SPACE'
from nhp_mri_prep.utils.nextflow import load_config
config = load_config('${config_file}')
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
print(effective_output_space)
PYTHON_OUTPUT_SPACE
    )
    TEMPLATE_NAME=\$(echo "\$EFFECTIVE_OUTPUT_SPACE" | cut -d':' -f1)
    
    # Thread environment variables are set by Nextflow's beforeScript based on task.cpus
    # Python code reads OMP_NUM_THREADS from environment
    
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_registration
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.templates import resolve_template, space_label_for
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, save_metadata, create_output_link
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path
import shutil
import os

# Load config
config = load_config('${config_file}')

# Get BIDS naming template (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Determine modality from BIDS naming template filename
modality = detect_modality(bids_name)

# Get effective_output_space from effective config file
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
template_name = space_label_for(effective_output_space)

# Resolve template
template_file = Path(resolve_template(effective_output_space))

# Check if unskullstripped version is provided and different from input file
unskullstripped_path = Path('${unskullstripped_file}')
input_path = Path('${input_file}')
use_unskullstripped = (unskullstripped_path.exists() and 
                       unskullstripped_path.stat().st_size > 0 and
                       str(unskullstripped_path) != str(input_path))

# Create step input for registration (use skullstripped version for computing transform)
input_obj = StepInput(
    input_file=input_path,
    working_dir=Path('work'),
    config=config,
    output_name='anat_registered.nii.gz',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}'
    }
)

# Run step to compute transform (on skullstripped version)
result = anat_registration(input_obj, template_file=template_file, template_name=template_name)

# If unskullstripped version is provided and different, apply transform to it instead
if use_unskullstripped:
    from nhp_mri_prep.operations.registration import ants_apply_transforms
    
    # Get the forward transform file
    forward_transform = result.additional_files.get("forward_transform")
    if not forward_transform or not forward_transform.exists():
        raise FileNotFoundError(f"Forward transform file not found: {forward_transform}")
    
    # Apply the transform to the unskullstripped version
    interpolation = config.get("registration", {}).get("interpolation", "BSpline")
    apply_result = ants_apply_transforms(
        movingf=str(unskullstripped_path),
        moving_type=0,  # 0: scalar (anatomical image)
        interpolation=interpolation,
        outputf_name='anat_registered.nii.gz',
        fixedf=str(template_file),
        working_dir=str(Path('work')),
        transformf=[str(forward_transform)],
        reff=str(template_file),
        logger=None
    )
    
    # Update the output file to point to the unskullstripped registered version
    result.output_file = Path(apply_result["imagef_registered"])

# Generate BIDS-compliant output filename with space entity
# Format: space-{template}_desc-preproc_{modality}.nii.gz
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix=f'space-{template_name}_desc-preproc',
    modality=modality
)

# Use symlink to avoid duplication - Nextflow publishDir will handle final copy
create_output_link(result.output_file, bids_output_filename)

# Sidecar for the registered image (template space -> TemplateSource auto-added from filename)
_reg_source = str(unskullstripped_path if use_unskullstripped else input_path)
write_derivative_sidecar(
    bids_output_filename,
    output_space=effective_output_space,
    resolved_template_path=str(template_file),
    skull_stripped=(not use_unskullstripped),
    sources=[_reg_source],
)

# Get filename stem for BIDS-compliant transform naming
original_stem = get_filename_stem(bids_name)

# Generate BIDS prefix (filename stem without modality)
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")

# Create symlinks for transform files with BIDS-compliant names (keep nature suffix: .nii.gz or .h5)
def _xfm_ext(p):
    r = Path(p).resolve()
    return ''.join(r.suffixes) if r.suffixes else r.suffix
if "forward_transform" in result.additional_files:
    ext = _xfm_ext(result.additional_files["forward_transform"])
    bids_transform_name = f"{bids_prefix_wo_modality}_from-{modality}_to-{template_name}_mode-image_xfm{ext}"
    create_output_link(result.additional_files["forward_transform"], bids_transform_name)
    write_derivative_sidecar(
        bids_transform_name,
        output_space=effective_output_space,
        resolved_template_path=str(template_file),
        engine=result.metadata.get('engine'),
        sources=[str(input_path), str(template_file)],
    )
if "inverse_transform" in result.additional_files:
    ext = _xfm_ext(result.additional_files["inverse_transform"])
    bids_transform_name = f"{bids_prefix_wo_modality}_from-{template_name}_to-{modality}_mode-image_xfm{ext}"
    create_output_link(result.additional_files["inverse_transform"], bids_transform_name)
    write_derivative_sidecar(
        bids_transform_name,
        output_space=effective_output_space,
        resolved_template_path=str(template_file),
        engine=result.metadata.get('engine'),
        sources=[str(input_path), str(template_file)],
    )

# Create ref_from_anat_reg.nii.gz for QC reference
# This is the template file used as the reference for registration
ref_from_anat_reg_path = Path(f"{bids_prefix_wo_modality}_ref_from_anat_reg.nii.gz")

# Create symlink to template file (the actual template used during registration)
# Use symlink - keep as symlink until published - saves storage
if template_file.exists():
    create_output_link(template_file, str(ref_from_anat_reg_path))
else:
    raise FileNotFoundError(f"Template file not found: {template_file}")

# Ensure file exists
if not ref_from_anat_reg_path.exists():
    raise FileNotFoundError(f"Failed to create ref_from_anat_reg.nii.gz: {ref_from_anat_reg_path}")

# Save metadata
save_metadata(result.metadata)
EOF
    """
}


process ANAT_BACKPROJECT_ATLASES_TO_T1W {
    label 'cpu'
    tag "${subject_id}_${session_id}"

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat/atlas_space-T1w",
        mode: 'copy',
        pattern: 'atlas/*.{nii.gz,json,tsv,md,bib}',
        saveAs: { f -> new File(f.toString()).name }

    input:
    tuple val(subject_id), val(session_id), val(bids_name), path(inverse_xfm), path(t1w_reference)
    path config_file

    output:
    // arity '0..*' allows zero matches (custom templates have no bundled atlases);
    // unlike a tuple `path(..., optional: true)`, which still errors on zero files.
    tuple val(subject_id), val(session_id), path("atlas/*.nii.gz", arity: '0..*'), val(bids_name), emit: output
    // Associated sidecars (LUT/docs/refs + JSON provenance) must be DECLARED outputs to be
    // published — publishDir only publishes declared outputs, not arbitrary work-dir files.
    // Kept out of `emit: output` so they never flow into scanner/surface or the ARM6 filter.
    path "atlas/*.{json,tsv,md,bib}", arity: '0..*', emit: sidecars

    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_backproject_atlases
from nhp_mri_prep.utils.nextflow import load_config
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path

config = load_config('${config_file}')
result = anat_backproject_atlases(
    inverse_xfm=Path('${inverse_xfm}'),
    t1w_reference=Path('${t1w_reference}'),
    bids_name=Path('${bids_name}'),
    working_dir=Path('.'),
    config=config,
)
# Outputs written to ./atlas/ (task work dir); atlases are ROIs in native T1w space
for _atlas in Path('atlas').glob('*.nii.gz'):
    write_derivative_sidecar(
        _atlas,
        roi_type="ROI",
        sources=[str(Path('${inverse_xfm}')), str(Path('${t1w_reference}'))],
    )
EOF
    """
}

process ANAT_BACKPROJECT_ATLASES_TO_SCANNER {
    label 'cpu'
    tag "${subject_id}_${session_id}"

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat/atlas_space-scanner",
        mode: 'copy',
        pattern: 'atlas/*.{nii.gz,json,tsv,md,bib}',
        saveAs: { f -> new File(f.toString()).name }

    input:
    tuple val(subject_id), val(session_id), val(bids_name), path(t1w_atlas_files, stageAs: 'atlas_input/*'), path(conform_inverse_xfm), path(scanner_reference)
    path config_file

    output:
    // arity '0..*' allows zero matches when upstream T1w atlas backprojection has no atlases
    // (custom templates); a tuple `path(..., optional: true)` still errors on zero files.
    tuple val(subject_id), val(session_id), path("atlas/*.nii.gz", arity: '0..*'), val(bids_name), emit: output
    // Sidecars are space-independent; carried into scanner space and DECLARED so publishDir
    // publishes them (declared outputs only). Separate from `emit: output` on purpose.
    path "atlas/*.{json,tsv,md,bib}", arity: '0..*', emit: sidecars

    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_reproject_atlases_to_scanner
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path

atlas_files = sorted(Path('atlas_input').glob('*.nii.gz'))
result = anat_reproject_atlases_to_scanner(
    atlas_files=atlas_files,
    conform_inverse_xfm=Path('${conform_inverse_xfm}'),
    scanner_reference=Path('${scanner_reference}'),
    working_dir=Path('.'),
)
# Outputs written to ./atlas/ (task work dir); atlases are ROIs in native scanner space
for _atlas in Path('atlas').glob('*.nii.gz'):
    write_derivative_sidecar(
        _atlas,
        roi_type="ROI",
        sources=[str(Path('${conform_inverse_xfm}')), str(Path('${scanner_reference}'))],
    )
EOF
    """
}

process ANAT_PROJECT_ATLASES_TO_SURFACE {
    label 'cpu'
    tag "${subject_id}_${session_id}"

    // Surface projection only runs when surf recon succeeded; a subject whose FastSurfer
    // outputs are missing simply drops out (mirrors ANAT_SURFACE_RECONSTRUCTION).
    errorStrategy 'ignore'

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat/atlas_space-fsnative",
        mode: 'copy',
        pattern: 'atlas/*.{nii.gz,func.gii,json,tsv,md,bib}',
        saveAs: { f -> new File(f.toString()).name }

    input:
    // fs_subject_dir is the FastSurfer subject tree staged as a task input (not read from
    // output_dir), so Nextflow guarantees it is present and complete before we run.
    tuple val(subject_id), val(session_id), val(bids_name), path(t1w_atlas_files, stageAs: 'atlas_input/*'), path(fs_subject_dir)
    path config_file

    output:
    // arity '0..*' allows zero matches (custom templates have no bundled atlases, or the
    // subject had no surface projection); unlike a tuple `path(..., optional: true)`, which
    // still errors on zero files and would drop the subject under errorStrategy 'ignore'.
    tuple val(subject_id), val(session_id), path("atlas/*.{nii.gz,func.gii}", arity: '0..*'), val(bids_name), emit: output
    // Sidecars + status/provenance JSON must be DECLARED to be published (publishDir only
    // publishes declared outputs). Separate emit so they don't affect downstream consumers.
    path "atlas/*.{json,tsv,md,bib}", arity: '0..*', emit: sidecars

    script:
    """
    \${PYTHON:-python3} <<EOF
import json
from nhp_mri_prep.steps.anatomical import (
    anat_project_atlases_to_surface,
    _atlas_name_from_filename,
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path

atlas_files = sorted(Path('atlas_input').glob('*.nii.gz'))

# fs_subject_dir is staged into the work dir; the function skips cleanly if it is empty.
result = anat_project_atlases_to_surface(
    atlas_files=atlas_files,
    fs_subject_dir=Path('${fs_subject_dir}'),
    bids_name=Path('${bids_name}'),
    working_dir=Path('.'),
)

# Provenance: map each output back to the T1w-space atlas its labels came from (staged
# basename, portable), matched by the atlas- entity. The fsnative grid is a resample target,
# not the label source, so it is not recorded as a source.
atlas_input_by_name = {
    _atlas_name_from_filename(p.name): p.name for p in atlas_files
}
# Outputs written to ./atlas/ (task work dir): fsnative-space ROI volumes + per-hemi surfaces
for _out in list(Path('atlas').glob('*.nii.gz')) + list(Path('atlas').glob('*.func.gii')):
    _src = atlas_input_by_name.get(_atlas_name_from_filename(_out.name))
    write_derivative_sidecar(_out, roi_type="ROI", sources=[_src] if _src else [])

# Always publish a status record so a silent skip (errorStrategy 'ignore' + arity '0..*')
# is distinguishable from a template that legitimately had no atlases.
Path('atlas').mkdir(parents=True, exist_ok=True)
(Path('atlas') / 'desc-atlasSurfaceProjection_status.json').write_text(
    json.dumps(result.metadata, indent=2)
)
EOF
    """
}

process ANAT_T2W_TO_T1W_REGISTRATION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*.{nii.gz,h5,mat,json}',
        saveAs: { filename -> filename == 'metadata.json' ? null : filename }

    input:
    tuple val(subject_id), val(session_id), path(t2w_file), val(bids_name)
    path(t1w_reference)
    path config_file
    
    output:
    tuple val(subject_id), val(session_id), path("*.nii.gz"), val(bids_name), emit: output
    // Transforms: [sub, ses, forward_transform, inverse_transform]
    // Note: T1w is in scanner (pre-conformed) space at registration time
    tuple val(subject_id), val(session_id), path("*from-T2wScanner_to-scanner_mode-image_xfm.{h5,mat,nii.gz}"), path("*from-scanner_to-T2wScanner_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    path "*.json", emit: metadata
    
    script:
    """
    # Thread environment variables are set by Nextflow's beforeScript based on task.cpus
    # Python code reads OMP_NUM_THREADS from environment
    
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.anatomical import anat_t2w_to_t1w_registration
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, save_metadata, create_output_link
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path
import shutil
import os

# Load config
config = load_config('${config_file}')

# Get BIDS naming template (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Determine modality from BIDS naming template filename
modality = detect_modality(bids_name)

# Get filename stem for BIDS-compliant transform naming
original_stem = get_filename_stem(bids_name)

# Get T1w reference file
t1w_reference = Path('${t1w_reference}')

# Create step input
input_obj = StepInput(
    input_file=Path('${t2w_file}'),
    working_dir=Path('work'),
    config=config,
    output_name='t2w_to_t1w_registered.nii.gz',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}'
    }
)

# Run step
result = anat_t2w_to_t1w_registration(input_obj, t1w_reference=t1w_reference)

# Generate BIDS-compliant output filename with space-scanner entity
# Format: space-scanner_T2w.nii.gz (after T2w→T1w registration)
# Note: T2w is registered to T1w in scanner (pre-conformed) space; same sense as from-scanner_to-T1w
# Only after applying conform transform is T2w in the preprocessed T1w space (space-T1w)
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='space-scanner',
    modality=modality
)

# Use symlink to avoid duplication - Nextflow publishDir will handle final copy
create_output_link(result.output_file, bids_output_filename)

# Sidecar (registered T2w in scanner space -> native, no template block)
write_derivative_sidecar(
    bids_output_filename,
    sources=[str(Path('${t2w_file}')), str(Path('${t1w_reference}'))],
)

# Generate BIDS prefix (filename stem without modality)
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")

# Create symlinks for transform files with BIDS-compliant names (keep nature suffix: .nii.gz or .h5)
# Note: T1w is in scanner (pre-conformed) space at this point
def _xfm_ext(p):
    r = Path(p).resolve()
    return ''.join(r.suffixes) if r.suffixes else r.suffix
if "forward_transform" in result.additional_files:
    ext = _xfm_ext(result.additional_files["forward_transform"])
    bids_transform_name = f"{bids_prefix_wo_modality}_from-T2wScanner_to-scanner_mode-image_xfm{ext}"
    create_output_link(result.additional_files["forward_transform"], bids_transform_name)
    write_derivative_sidecar(bids_transform_name, engine=result.metadata.get('engine'), sources=[str(Path('${t2w_file}')), str(Path('${t1w_reference}'))])
if "inverse_transform" in result.additional_files:
    ext = _xfm_ext(result.additional_files["inverse_transform"])
    bids_transform_name = f"{bids_prefix_wo_modality}_from-scanner_to-T2wScanner_mode-image_xfm{ext}"
    create_output_link(result.additional_files["inverse_transform"], bids_transform_name)
    write_derivative_sidecar(bids_transform_name, engine=result.metadata.get('engine'), sources=[str(Path('${t2w_file}')), str(Path('${t1w_reference}'))])

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

// ============================================
// PASS-THROUGH HELPER PROCESSES (for skipped steps)
// ============================================

process ANAT_CONFORM_PASSTHROUGH {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name)
    path config_file  // Effective config file with all resolved parameters
    
    output:
    // Exact token: must not also capture *_desc-conformFullFOV_*, or downstream steps
    // get a two-element list where they expect one file.
    tuple val(subject_id), val(session_id), path("*_desc-conform_*.nii.gz"), val(bids_name), emit: output
    // Transforms: [sub, ses, forward_transform, inverse_transform]
    tuple val(subject_id), val(session_id), path("*from-scanner_to-*_mode-image_xfm.{h5,mat,nii.gz}"), path("*from-*_to-scanner_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    // Reference: [sub, ses, reference]
    tuple val(subject_id), val(session_id), path("reference.nii.gz"), emit: reference
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import detect_modality, save_metadata, create_output_link
from pathlib import Path
import shutil
import os
import numpy as np
import json

# Get BIDS naming template
bids_name = Path('${bids_name}')

# Determine modality
modality = detect_modality(bids_name)
original_stem = get_filename_stem(bids_name)

# Pass through input file (create symlink)
# Use create_output_link() for consistency and proper symlink resolution
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-conform',
    modality=modality
)
create_output_link(Path('${input_file}'), bids_output_filename)

# Generate BIDS prefix
bids_prefix = original_stem.replace(f"_{modality}", "")

# Create identity transform matrices (FLIRT format: 4x4 matrix)
# Identity matrix: 1.0 on diagonal, 0.0 elsewhere
identity_matrix = np.eye(4)

# Forward transform: from-scanner_to-{modality}
forward_transform_name = f"{bids_prefix}_from-scanner_to-{modality}_mode-image_xfm.mat"
np.savetxt(forward_transform_name, identity_matrix, fmt='%.6f', delimiter=' ')

# Inverse transform: from-{modality}_to-scanner
inverse_transform_name = f"{bids_prefix}_from-{modality}_to-scanner_mode-image_xfm.mat"
np.savetxt(inverse_transform_name, identity_matrix, fmt='%.6f', delimiter=' ')

# Create dummy reference (symlink to input for QC compatibility)
# Keep as symlink until published - saves storage
reference_name = 'reference.nii.gz'
create_output_link(Path('${input_file}'), reference_name)

# Save metadata
metadata = {
    "step": "conform",
    "skipped": True,
    "reason": "disabled in configuration",
    "modality": modality
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
EOF
    """
}

process ANAT_BIAS_CORRECTION_PASSTHROUGH {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name)
    path config_file
    
    output:
    // Bias-corrected full head output (passthrough): [sub, ses, biascorrected_file, bids_template]
    // Use T?w pattern to match both T1w and T2w modalities
    tuple val(subject_id), val(session_id), path("*desc-biascorrect*_T?w.nii.gz"), val(bids_name), emit: output
    // Brain: same file as output when step skipped (for registration when anat is already skullstripped)
    // Brain output: [sub, ses, brain_file] - will be joined with bids_template in workflow
    // Use desc-biascorrect naming (not desc-preproc) - publishing step will handle preproc naming
    // Use T?w pattern to match both T1w and T2w modalities
    tuple val(subject_id), val(session_id), path("*desc-biascorrect*_T?w_brain.nii.gz"), emit: brain
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import detect_modality, save_metadata, create_output_link
from pathlib import Path
import os

# Get BIDS naming template
bids_name = Path('${bids_name}')

# Determine modality
modality = detect_modality(bids_name)
original_stem = get_filename_stem(bids_name)

# Pass through input file (create symlink)
# Use create_output_link() for consistency and proper symlink resolution
# Rename to desc-biascorrect for pipeline consistency even when step is disabled
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-biascorrect',
    modality=modality
)
create_output_link(Path('${input_file}'), bids_output_filename)

# Use same file as output for brain (no separate brain when step is skipped).
# Registration and downstream can use this as the moving/reference image.
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")
bids_brain_filename = f"{bids_prefix_wo_modality}_desc-biascorrect_{modality}_brain.nii.gz"
create_output_link(Path('${input_file}'), bids_brain_filename)

# Save metadata
metadata = {
    "step": "bias_correction",
    "skipped": True,
    "reason": "disabled in configuration",
    "modality": modality
}
save_metadata(metadata)
EOF
    """
}

process ANAT_REGISTRATION_PASSTHROUGH {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), path(input_file), val(bids_name)
    path config_file  // Effective config file with all resolved parameters
    
    output:
    // Output: [sub, ses, registered_file, bids_template]
    tuple val(subject_id), val(session_id), path("*.nii.gz"), val(bids_name), emit: output
    // Transforms: [sub, ses, bids_name, forward_transform, inverse_transform]
    // Forward: from-{modality}_to-{template} (e.g., from-T1w_to-NMT2Sym)
    // Inverse: from-{template}_to-{modality} (e.g., from-NMT2Sym_to-T1w)
    // Use patterns that match the actual naming: forward starts with from-T1w_to- or from-T2w_to-, inverse ends with _to-T1w or _to-T2w
    tuple val(subject_id), val(session_id), val(bids_name), path("*from-T1w_to-*_mode-image_xfm.{h5,mat,nii.gz}"), path("*from-*_to-T1w_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    // Reference: [sub, ses, reference_file]
    tuple val(subject_id), val(session_id), path("*ref_from_anat_reg.nii.gz"), emit: reference
    path "*.json", emit: metadata
    
    script:
    """
    # Get effective_output_space from effective config file
    EFFECTIVE_OUTPUT_SPACE=\$(\${PYTHON:-python3} <<'PYTHON_OUTPUT_SPACE'
from nhp_mri_prep.utils.nextflow import load_config
config = load_config('${config_file}')
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
print(effective_output_space)
PYTHON_OUTPUT_SPACE
    )
    TEMPLATE_NAME=\$(echo "\$EFFECTIVE_OUTPUT_SPACE" | cut -d':' -f1)
    
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.templates import resolve_template, space_label_for
from nhp_mri_prep.utils.nextflow import detect_modality, save_metadata, create_output_link, load_config
from pathlib import Path
import os
import subprocess
import shutil

# Get BIDS naming template
bids_name = Path('${bids_name}')

# Determine modality
modality = detect_modality(bids_name)
original_stem = get_filename_stem(bids_name)

# Get effective_output_space from effective config file
config = load_config('${config_file}')
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
template_name = space_label_for(effective_output_space)

# Pass through input file (create symlink with space entity)
# Use create_output_link() for consistency and proper symlink resolution
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix=f'space-{template_name}_desc-preproc',
    modality=modality
)
create_output_link(Path('${input_file}'), bids_output_filename)

# Generate BIDS prefix
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")

# Get effective_output_space from effective config file
config = load_config('${config_file}')
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')

# Create identity transform using ANTs
# Create identity affine transform file first
template_file = Path(resolve_template(effective_output_space))
input_file = Path('${input_file}')

# Forward transform: from-{modality}_to-{template_name}
forward_transform_name = f"{bids_prefix_wo_modality}_from-{modality}_to-{template_name}_mode-image_xfm.h5"

# Create identity affine transform (.txt format)
identity_affine = Path('identity_affine.txt')
with open(identity_affine, 'w') as f:
    f.write('#Insight Transform File V1.0\n')
    f.write('#Transform 0\n')
    f.write('Transform: AffineTransform_double_3_3\n')
    f.write('Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n')
    f.write('FixedParameters: 0 0 0\n')

# Convert to .h5 using ConvertTransformFile
try:
    cmd_convert = [
        'ConvertTransformFile', '3',
        str(identity_affine),
        forward_transform_name
    ]
    result = subprocess.run(cmd_convert, check=True, capture_output=True, text=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    # If ConvertTransformFile fails, create minimal .h5 structure
    # This is a simplified approach - proper identity would need full ANTs transform structure
    with open(forward_transform_name, 'w') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('#Transform 0\n')
        f.write('Transform: AffineTransform_double_3_3\n')
        f.write('Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n')
        f.write('FixedParameters: 0 0 0\n')

# Inverse transform: from-{template_name}_to-{modality} (same as forward for identity)
# Use symlink since it's the same file - saves storage
inverse_transform_name = f"{bids_prefix_wo_modality}_from-{template_name}_to-{modality}_mode-image_xfm.h5"
create_output_link(forward_transform_name, inverse_transform_name)

# Create ref_from_anat_reg.nii.gz for QC reference (same as in ANAT_REGISTRATION) 
# Use symlink - keep as symlink until published - saves storage
ref_from_anat_reg_path = Path(f"{bids_prefix_wo_modality}_ref_from_anat_reg.nii.gz")

# Create symlink to template file
if template_file.exists():
    create_output_link(template_file, str(ref_from_anat_reg_path))
else:
    raise FileNotFoundError(f"Template file not found: {template_file}")

# Ensure file exists
if not ref_from_anat_reg_path.exists():
    raise FileNotFoundError(f"Failed to create ref_from_anat_reg.nii.gz: {ref_from_anat_reg_path}")

# Save metadata
metadata = {
    "step": "registration",
    "skipped": True,
    "reason": "disabled in configuration",
    "modality": modality,
    "target": template_name
}
save_metadata(metadata)
EOF
    """
}

// ============================================
// ANATOMICAL APPLY MODULES (for T2w processing)
// ============================================
// These modules apply T1w's computed transforms/masks to T2w data

process ANAT_APPLY_CONFORM {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*desc-preproc*.{nii.gz,json}'

    input:
    // Input: [sub, ses, t2w_file, t2w_bids_name, conform_transform, conformed_reference, anat_ses]
    // Stage conformed_reference as reg_reference.nii.gz to avoid output pattern collision
    tuple val(subject_id), val(session_id), path(t2w_file), val(bids_name), path(conform_transform), path(conformed_reference, stageAs: 'reg_reference.nii.gz'), val(anatomical_session)
    path config_file

    output:
    // Output: [sub, ses, conformed_t2w, t2w_bids_name, anat_ses]
    tuple val(subject_id), val(session_id), path("*desc-conform_T2w.nii.gz"), val(bids_name), val(anatomical_session), emit: output
    // Phase 1 preproc output: [sub, ses, preproc_t2w, t2w_bids_name, anat_ses]
    // T2w is in preprocessed T1w space (after applying conform transform)
    tuple val(subject_id), val(session_id), path("*space-T1w_desc-preproc_T2w.nii.gz"), val(bids_name), val(anatomical_session), emit: preproc_output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.operations.registration import flirt_apply_transforms
from nhp_mri_prep.utils.bids import create_bids_output_filename
from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata, load_config
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path

# Load config
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Apply conform transform to T2w
t2w_result = flirt_apply_transforms(
    movingf=str(Path('${t2w_file}')),
    outputf_name='t2w_conformed.nii.gz',
    reff=str(Path('reg_reference.nii.gz')),
    working_dir='work',
    transformf=str(Path('${conform_transform}')),
    logger=None,
    interpolation='trilinear',
    generate_tmean=False
)

if not t2w_result.get("imagef_registered"):
    raise FileNotFoundError("Failed to apply conform transform to T2w")

# Generate BIDS-compliant output filename (for internal workflow use)
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-conform',
    modality='T2w'
)

# Create symlink (for internal workflow)
create_output_link(Path(t2w_result["imagef_registered"]), bids_output_filename)

# Generate BIDS-compliant output filename for Phase 1 preproc (published output)
# T2w is now in preprocessed T1w space (after applying conform transform)
bids_preproc_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='space-T1w_desc-preproc',
    modality='T2w'
)

# Create symlink for Phase 1 preproc output (published)
# Use the same source file as desc-conform (t2w_result["imagef_registered"])
t2w_source = Path(t2w_result["imagef_registered"])
if not t2w_source.exists():
    raise FileNotFoundError(f"Source file for preproc output does not exist: {t2w_source}")

create_output_link(t2w_source, bids_preproc_filename)

# Sidecar for preproc T2w (native space-T1w -> no template block)
write_derivative_sidecar(
    bids_preproc_filename,
    sources=[str(Path('${t2w_file}')), str(Path('${conform_transform}'))],
)

# Verify the file exists and is accessible (Nextflow needs to see it)
preproc_path = Path(bids_preproc_filename)
if not preproc_path.exists():
    raise FileNotFoundError(f"Failed to create preproc output file: {bids_preproc_filename} (source: {t2w_source})")
# Check if it's a valid symlink or file
if not (preproc_path.is_symlink() or preproc_path.is_file()):
    raise FileNotFoundError(f"Preproc output file exists but is not a valid file or symlink: {bids_preproc_filename}")
# If it's a symlink, verify it resolves correctly
if preproc_path.is_symlink():
    try:
        resolved = preproc_path.resolve(strict=True)
        if not resolved.exists():
            raise FileNotFoundError(f"Preproc symlink points to non-existent file: {bids_preproc_filename} -> {resolved}")
    except Exception as e:
        raise FileNotFoundError(f"Preproc symlink is broken: {bids_preproc_filename}, error: {e}")

# Save metadata
save_metadata({
    "step": "apply_conform",
    "modality": "T2w",
    "t2w_file": str(Path('${t2w_file}')),
    "conform_transform": str(Path('${conform_transform}')),
    "anatomical_session": '${anatomical_session}'
})
EOF
    """
}

process ANAT_APPLY_TRANSFORMATION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*.{nii.gz,h5,json}',
        saveAs: { filename -> (filename.contains('target_final.nii.gz') || filename == 'metadata.json') ? null : filename }

    input:
    // Input: [sub, ses, masked_t2w, t2w_bids_name, registration_transform, registration_reference]
    tuple val(subject_id), val(session_id), path(masked_t2w), val(bids_name), path(registration_transform), path(registration_reference)
    path config_file
    
    output:
    // Output: [sub, ses, registered_t2w, t2w_bids_name]
    // Pattern excludes target_final files (they have target_final before .nii.gz)
    tuple val(subject_id), val(session_id), path("*space-*desc-preproc*T2w.nii.gz"), val(bids_name), emit: output
    // Reference file for QC: final target reference at appropriate resolution
    tuple val(subject_id), val(session_id), path("*target_final.nii.gz"), emit: reference
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.operations.registration import ants_apply_transforms
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata, load_config
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from nhp_mri_prep.utils.templates import resolve_template
from pathlib import Path
import re
import shutil

# Load config
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Determine output space from transform filename
transform_path = Path('${registration_transform}')
transform_stem = get_filename_stem(transform_path)

# Extract template name from transform filename (e.g., "from-T1w_to-NMT2Sym" -> "NMT2Sym")
space_match = re.search(r'to-([A-Za-z0-9]+)', transform_stem)
if space_match:
    template_name = space_match.group(1)
else:
    # Fallback: get from config
    template_name = config.get('template', {}).get('output_space', 'NMT2Sym:res-05').split(':')[0]

# Apply registration transform to T2w
interpolation = config.get("registration", {}).get("interpolation", "BSpline")
t2w_result = ants_apply_transforms(
    movingf=str(Path('${masked_t2w}')),
    moving_type=0,  # 0: scalar (anatomical image)
    interpolation=interpolation,
    outputf_name='t2w_registered.nii.gz',
    fixedf=str(Path('${registration_reference}')),
    working_dir='work',
    transformf=[str(Path('${registration_transform}'))],
    reff=str(Path('${registration_reference}')),
    logger=None
)

if not t2w_result.get("imagef_registered"):
    raise FileNotFoundError("Failed to apply registration transform to T2w")

# Generate BIDS-compliant output filename with space entity
bids_output_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix=f'space-{template_name}_desc-preproc',
    modality='T2w'
)

# Create symlink
create_output_link(Path(t2w_result["imagef_registered"]), bids_output_filename)

# Sidecar (T2w in template space -> TemplateSource from config output_space)
_apply_output_space = config.get('template', {}).get('output_space')
try:
    _apply_template_path = str(resolve_template(_apply_output_space)) if _apply_output_space else None
except Exception:
    _apply_template_path = None
write_derivative_sidecar(
    bids_output_filename,
    output_space=_apply_output_space,
    resolved_template_path=_apply_template_path,
    sources=[str(Path('${masked_t2w}')), str(Path('${registration_transform}')), str(Path('${registration_reference}'))],
)

# Create reference file for QC (copy of registration_reference)
ref_from_reg_path = Path(f"{get_filename_stem(bids_output_filename)}_target_final.nii.gz")
shutil.copy2(Path('${registration_reference}'), ref_from_reg_path)

# Save metadata
save_metadata({
    "step": "apply_registration",
    "modality": "T2w",
    "t2w_file": str(Path('${masked_t2w}')),
    "registration_transform": str(Path('${registration_transform}')),
    "template_name": template_name
})
EOF
    """
}

process ANAT_APPLY_TRANSFORM_MASK {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*desc-brain_mask.{nii.gz,json}'

    input:
    // Input: [sub, ses, mask_file, registration_transform, registration_reference]
    tuple val(subject_id), val(session_id), path(mask_file), path(registration_transform), path(registration_reference)
    path config_file

    output:
    // Output: [sub, ses, transformed_mask]
    tuple val(subject_id), val(session_id), path("*space-*desc-brain_mask.nii.gz"), emit: output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.operations.registration import ants_apply_transforms
from nhp_mri_prep.utils.bids import get_filename_stem, replace_bids_space
from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata, load_config
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from nhp_mri_prep.utils.templates import resolve_template
from pathlib import Path
import re

# Load config
config = load_config('${config_file}')

# Get mask file path
mask_file = Path('${mask_file}')

# Determine output space from transform filename
transform_path = Path('${registration_transform}')
transform_stem = get_filename_stem(transform_path)

# Extract template name from transform filename (e.g., "from-T1w_to-NMT2Sym" -> "NMT2Sym")
space_match = re.search(r'to-([A-Za-z0-9]+)', transform_stem)
if space_match:
    template_name = space_match.group(1)
else:
    # Fallback: get from config
    template_name = config.get('template', {}).get('output_space', 'NMT2Sym:res-05').split(':')[0]

# Apply registration transform to mask using NearestNeighbor interpolation (for binary masks)
mask_result = ants_apply_transforms(
    movingf=str(mask_file),
    moving_type=0,  # 0: scalar
    interpolation='NearestNeighbor',  # NearestNeighbor for binary masks
    outputf_name='mask_registered.nii.gz',
    fixedf=str(Path('${registration_reference}')),
    working_dir='work',
    transformf=[str(transform_path)],
    reff=str(Path('${registration_reference}')),
    logger=None,
    generate_tmean=False
)

if not mask_result.get("imagef_registered"):
    raise FileNotFoundError("Failed to apply registration transform to mask")

# Replace space entity on mask stem (strip any existing space-* first)
mask_stem = get_filename_stem(mask_file)
bids_output_filename = f"{replace_bids_space(mask_stem, template_name)}.nii.gz"

# Create symlink
create_output_link(Path(mask_result["imagef_registered"]), bids_output_filename)

# Sidecar (brain mask in template space -> TemplateSource from config output_space)
_mask_output_space = config.get('template', {}).get('output_space')
try:
    _mask_template_path = str(resolve_template(_mask_output_space)) if _mask_output_space else None
except Exception:
    _mask_template_path = None
write_derivative_sidecar(
    bids_output_filename,
    output_space=_mask_output_space,
    resolved_template_path=_mask_template_path,
    roi_type="Brain",
    sources=[str(mask_file), str(transform_path), str(Path('${registration_reference}'))],
)

# Save metadata
save_metadata({
    "step": "apply_mask_transform",
    "mask_file": str(mask_file),
    "registration_transform": str(transform_path),
    "template_name": template_name,
    "interpolation": "NearestNeighbor"
})
EOF
    """
}

process ANAT_PUBLISH_PHASE1 {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*desc-preproc*.{nii.gz,json}',
        saveAs: { filename ->
            // Only publish files matching desc-preproc pattern (excludes the generic metadata.json)
            if (filename.contains('desc-preproc')) {
                return filename
            }
            return null
        }

    input:
    tuple val(subject_id), val(session_id), path(anat_file), path(brain_file), val(bids_name)
    path config_file

    output:
    tuple val(subject_id), val(session_id), path("*desc-preproc_T*w.nii.gz"), val(bids_name), emit: output
    tuple val(subject_id), val(session_id), path("*desc-preproc_*_brain.nii.gz", arity: '0..*'), emit: brain
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<EOF
from pathlib import Path
import re
import sys
from nhp_mri_prep.utils.nextflow import save_metadata, create_output_link
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar

# Get input files
anat_file = Path('${anat_file}')
brain_file = Path('${brain_file}')

# Function to replace desc-{stepname} with desc-preproc in filename
def replace_desc_with_preproc(filename):
    # [^_]+ stops at the first underscore so the modality token (e.g., _T1w, _T2w)
    # is preserved. Using \\w+ would greedily consume past the underscore and drop modality.
    pattern = r'desc-[^_]+_'
    replacement = 'space-T1w_desc-preproc_'
    result = re.sub(pattern, replacement, str(filename))
    return Path(result)

_VALID_PREPROC_SUFFIXES = (
    '_T1w.nii.gz', '_T2w.nii.gz',
    '_T1w_brain.nii.gz', '_T2w_brain.nii.gz',
    '_space-T1w_desc-preproc_T1w.nii.gz',
    '_space-T1w_desc-preproc_T1w_brain.nii.gz',
)

def validate_preproc_modality(name, source):
    name_str = str(name)
    if not any(name_str.endswith(s) for s in _VALID_PREPROC_SUFFIXES):
        raise ValueError(
            f"Published name '{name_str}' (from {source}) missing modality token; "
            f"expected suffix in {_VALID_PREPROC_SUFFIXES}"
        )

# Track created files for output declaration
created_files = []

# Process anat_file (full head) if not dummy
if '.dummy' not in str(anat_file):
    anat_preproc_filename = replace_desc_with_preproc(anat_file.name)
    validate_preproc_modality(anat_preproc_filename, anat_file)
    create_output_link(anat_file, anat_preproc_filename)
    # Verify the symlink was created and is accessible
    if not Path(anat_preproc_filename).exists():
        raise FileNotFoundError(f"Failed to create symlink: {anat_preproc_filename}")
    # Sidecar (preproc full head, native T1w space, not skull-stripped)
    write_derivative_sidecar(anat_preproc_filename, skull_stripped=False, sources=[str(anat_file)])
    created_files.append(anat_preproc_filename)
    print(f"Created symlink: {anat_preproc_filename} -> {anat_file}", file=sys.stderr)
    print(f"Symlink exists: {Path(anat_preproc_filename).exists()}, is_symlink: {Path(anat_preproc_filename).is_symlink()}", file=sys.stderr)

# Process brain_file if not dummy
if '.dummy' not in str(brain_file):
    brain_preproc_filename = replace_desc_with_preproc(brain_file.name)
    validate_preproc_modality(brain_preproc_filename, brain_file)
    create_output_link(brain_file, brain_preproc_filename)
    # Verify the symlink was created and is accessible
    if not Path(brain_preproc_filename).exists():
        raise FileNotFoundError(f"Failed to create symlink: {brain_preproc_filename}")
    # Sidecar (preproc brain, native T1w space, skull-stripped)
    write_derivative_sidecar(brain_preproc_filename, skull_stripped=True, sources=[str(brain_file)])
    created_files.append(brain_preproc_filename)
    print(f"Created symlink: {brain_preproc_filename} -> {brain_file}", file=sys.stderr)
    print(f"Symlink exists: {Path(brain_preproc_filename).exists()}, is_symlink: {Path(brain_preproc_filename).is_symlink()}", file=sys.stderr)

# Print summary for debugging
print(f"Total files created for publishing: {len(created_files)}", file=sys.stderr)
for f in created_files:
    print(f"  - {f}", file=sys.stderr)

# Save metadata
save_metadata({
    "step": "publish_phase1",
    "subject_id": "${subject_id}",
    "session_id": "${session_id}" if "${session_id}" else None,
    "anat_file": str(anat_file),
    "brain_file": str(brain_file)
})
EOF
    """
}

process ANAT_T1WT2W_COMBINED {
    errorStrategy 'ignore'
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/anat",
        mode: 'copy',
        pattern: '*T1wT2wCombined*.{nii.gz,json}'

    input:
    tuple val(subject_id), val(session_id), path(t1w_file), val(t1w_bids_name), path(t2w_file), path(segmentation_file), path(segmentation_lut)
    path config_file

    output:
    tuple val(subject_id), val(session_id), path("*T1wT2wCombined.nii.gz"), val(t1w_bids_name), emit: output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<EOF
from pathlib import Path
from nhp_mri_prep.steps.anatomical import anat_t1wt2wcombined
from nhp_mri_prep.utils.nextflow import save_metadata, create_output_link
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
# Get input files
t1w_file = Path('${t1w_file}')
t2w_file = Path('${t2w_file}')
segmentation_file = Path('${segmentation_file}')
segmentation_lut_file = Path('${segmentation_lut}')
t1w_bids_name = Path('${t1w_bids_name}')

# t1w_file is Phase 1 preproc (*_space-T1w_desc-preproc_T1w.nii.gz); only change modality suffix
t1w_preproc_name = t1w_file.name
if not t1w_preproc_name.endswith('_T1w.nii.gz'):
    raise ValueError(
        f"Expected Phase 1 preproc T1w filename, got: {t1w_preproc_name}"
    )
output_filename = t1w_preproc_name.replace('_T1w.nii.gz', '_T1wT2wCombined.nii.gz')

# Call the step function
result = anat_t1wt2wcombined(
    t1w_file=t1w_file,
    t2w_file=t2w_file,
    segmentation_file=segmentation_file,
    segmentation_lut_file=segmentation_lut_file,
    output_file=Path(output_filename),
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}' if '${session_id}' else None
    }
)

# The function already saved the file with the correct name, so result.output_file
# should match output_filename. Verify and create link if needed for Nextflow pattern matching
if result.output_file.name != output_filename:
    create_output_link(result.output_file, output_filename)

# Sidecar (combined T1w/T2w, native space-T1w -> no template block); anchor on the
# published name since create_output_link above is conditional.
write_derivative_sidecar(
    output_filename,
    sources=[str(t1w_file), str(t2w_file), str(segmentation_file), str(segmentation_lut_file)],
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}
