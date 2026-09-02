/*
 * Quality control modules for nhp_mri_prep Nextflow pipeline
 */

// ============================================
// ANATOMICAL QC PROCESSES
// ============================================

process QC_CONFORM {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(conformed_file), val(bids_naming_template), path(template_resampled_file, stageAs: 'template.nii.gz'), path(full_fov_file, stageAs: 'full_fov.nii.gz'), path(full_fov_template_file, stageAs: 'full_fov_template.nii.gz'), path(fov_box_file, stageAs: 'fov_box.nii.gz')
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_conform
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Use the resampled template file from conform step (matches the conformed image space)
# File is staged as 'template.nii.gz' to avoid filename collisions
template_file = Path('template.nii.gz')

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-conform',
    modality=modality
).replace('.nii.gz', '.png')

# Second figure, on the uncropped grid, with the processing FOV drawn as a box.
qc_full_fov_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-conformFullFOV',
    modality=modality
).replace('.nii.gz', '.png')

def _real(p):
    # The full-FOV trio is an optional ANAT_CONFORM output; the workflow substitutes
    # empty .dummy sentinels when it is absent. stageAs rewrites the sentinel to the
    # real name, so the extension is gone by the time we see it -- emptiness is the
    # only signal left. All three are absent together, so any one of them decides.
    path = Path(p)
    if (not path.exists()) or path.stat().st_size == 0:
        return None
    return path

full_fov_file = _real('full_fov.nii.gz')
full_fov_template_file = _real('full_fov_template.nii.gz')
fov_box_file = _real('fov_box.nii.gz')

# Generate QC. The full-FOV trio is staged under fixed names to avoid collisions
# with the conformed image.
result = qc_conform(
    conformed_file=Path('${conformed_file}'),
    template_file=template_file,
    output_path=Path(qc_output_filename),
    modality='anat',
    config=config,
    full_fov_file=full_fov_file,
    full_fov_template_file=full_fov_template_file,
    fov_box_file=fov_box_file,
    full_fov_output_path=Path(qc_full_fov_filename),
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_BIAS_CORRECTION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(original_file), path(corrected_file), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_bias_correction
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-biascorrect',
    modality=modality
).replace('.nii.gz', '.png')

# Generate QC
result = qc_bias_correction(
    original_file=Path('${original_file}'),
    corrected_file=Path('${corrected_file}'),
    output_path=Path(qc_output_filename),
    modality='anat',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_T1WT2W_COMBINED {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(t1w_before_file), path(t1wt2w_combined_file), path(mask_file), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_t1wt2w_combined
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-T1wT2wCombined',
    modality=modality
).replace('.nii.gz', '.png')

# Get mask file (optional - may be dummy)
mask_file_path = Path('${mask_file}')
mask_file_optional = mask_file_path if mask_file_path.exists() and not str(mask_file_path).endswith('.dummy') else None

# Generate QC
result = qc_t1wt2w_combined(
    t1w_before_file=Path('${t1w_before_file}'),
    t1wt2w_combined_file=Path('${t1wt2w_combined_file}'),
    output_path=Path(qc_output_filename),
    modality='anat',
    mask_file=mask_file_optional,
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_SKULLSTRIPPING {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(brain_file), path(mask_file), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_skullstripping
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-skullstrip',
    modality=modality
).replace('.nii.gz', '.png')

# Generate QC
result = qc_skullstripping(
    underlay_file=Path('${brain_file}'),
    mask_file=Path('${mask_file}'),
    output_path=Path(qc_output_filename),
    modality='anat',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_ATLAS_SEGMENTATION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(brain_file), path(segmentation_file), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", optional: true, emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_atlas_segmentation
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Check if segmentation file exists
segmentation_file = Path('${segmentation_file}')
if not segmentation_file.exists():
    logger.warning(f"QC: Atlas segmentation file not found - {segmentation_file}. Skipping atlas segmentation QC.")
    # Create empty metadata file
    save_metadata({"step": "qc_atlas_segmentation", "skipped": True, "reason": "segmentation_file_not_found"})
else:
    # Generate BIDS-compliant QC output filename
    qc_output_filename = create_bids_output_filename(
        original_file_path=bids_naming_template,
        suffix='desc-atlasSegmentation',
        modality=modality
    ).replace('.nii.gz', '.png')
    
    # Generate QC
    result = qc_atlas_segmentation(
        underlay_file=Path('${brain_file}'),
        atlas_file=segmentation_file,
        output_path=Path(qc_output_filename),
        modality='anat',
        config=config
    )
    
    # Save metadata
    save_metadata(result.metadata)
EOF
    """
}

process QC_REGISTRATION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(registered_file, stageAs: 'registered.nii.gz'), path(reference_file, stageAs: 'reference.nii.gz'), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_registration
from nhp_mri_prep.utils.bids import parse_bids_entities, create_bids_filename, create_bids_output_filename
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get the registered file and reference file directly from input
# Files are staged with unique names to avoid collisions
registered_file = Path('registered.nii.gz')
if not registered_file.exists():
    raise FileNotFoundError(f"Registered file not found: {registered_file}")

reference_file = Path('reference.nii.gz')
if not reference_file.exists():
    raise FileNotFoundError(f"Reference file not found: {reference_file}")

# Get BIDS naming template (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from BIDS naming template filename
modality = detect_modality(bids_naming_template)

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-anat2template',
    modality=modality
).replace('.nii.gz', '.png')

# Generate QC
result = qc_registration(
    image_file=registered_file,
    template_file=reference_file,  # Use reference file from registration output
    output_path=Path(qc_output_filename),
    modality='anat2template',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_T2W_TO_T1W_REGISTRATION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(t2w_in_t1w_space), path(t1w_brain)
    val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_registration
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get T1w brain file (skullstripped T1w for contour overlay)
t1w_brain = Path('${t1w_brain}')
if not t1w_brain.exists():
    raise FileNotFoundError(f"T1w brain file not found: {t1w_brain}")

# Get conformed T2w (after applying conform transform)
conformed_t2w = Path('${t2w_in_t1w_space}')
if not conformed_t2w.exists():
    raise FileNotFoundError(f"Conformed T2w file not found: {conformed_t2w}")

# Get BIDS naming template (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from BIDS naming template filename
original_stem = get_filename_stem(bids_naming_template)
modality = 'T2w'

# Generate BIDS-compliant QC output filename
# Format: {prefix}_desc-T2w2T1w_T2w.png
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")
qc_output_filename = f"{bids_prefix_wo_modality}_desc-T2w2T1w_{modality}.png"

# Generate QC: Conformed T2w (underlay) with T1w brain contours (overlay)
result = qc_registration(
    image_file=conformed_t2w,
    template_file=t1w_brain,  # T1w brain for contour overlay
    output_path=Path(qc_output_filename),
    modality='T2w2T1w',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_T2W_TEMPLATE_SPACE {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(t2w_in_template_space), path(template_reference)
    val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_registration
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get template reference file (for contour overlay)
template_reference = Path('${template_reference}')
if not template_reference.exists():
    raise FileNotFoundError(f"Template reference file not found: {template_reference}")

# Get T2w in template space (after all transforms)
t2w_in_template_space = Path('${t2w_in_template_space}')
if not t2w_in_template_space.exists():
    raise FileNotFoundError(f"T2w in template space file not found: {t2w_in_template_space}")

# Get BIDS naming template (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from BIDS naming template filename
original_stem = get_filename_stem(bids_naming_template)
modality = 'T2w'

# Generate BIDS-compliant QC output filename
# Format: {prefix}_desc-T2w2template_T2w.png
bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")
qc_output_filename = f"{bids_prefix_wo_modality}_desc-T2w2template_{modality}.png"

# Generate QC: T2w in template space (underlay) with template reference contours (overlay)
result = qc_registration(
    image_file=t2w_in_template_space,
    template_file=template_reference,  # Template reference for contour overlay
    output_path=Path(qc_output_filename),
    modality='T2w2template',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_SURF_RECON_TISSUE_SEG {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(actual_subject_id), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_surf_recon_tissue_seg
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-surfReconTissueSeg',
    modality=modality
).replace('.nii.gz', '.png')

# Use the actual subject ID passed from ANAT_SURFACE_RECONSTRUCTION
# This is the real directory name (e.g., 'sub-032309m' or 'sub-032309m_ses-001')
actual_subject_id = '${actual_subject_id}'
published_base = Path('${params.output_dir}/fastsurfer')
fs_subject_dir_resolved = published_base / actual_subject_id

# Verify the directory exists
if not fs_subject_dir_resolved.exists():
    raise FileNotFoundError(
        f"FastSurfer subject directory not found: {fs_subject_dir_resolved}"
    )

# Generate QC
result = qc_surf_recon_tissue_seg(
    fs_subject_dir=fs_subject_dir_resolved,
    output_path=Path(qc_output_filename),
    modality='anat',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_CORTICAL_SURF_AND_MEASURES {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(actual_subject_id), val(bids_naming_template), val(atlas_name)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_cortical_surf_and_measures
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Determine modality from original filename
modality = detect_modality(bids_naming_template)

# Get atlas name (remove "atlas" suffix if present for compatibility)
atlas_name = '${atlas_name}'.rstrip('atlas') if '${atlas_name}'.endswith('atlas') else '${atlas_name}'
if not atlas_name:
    atlas_name = 'ARM2'  # default

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-corticalSurfAndMeasures',
    modality=modality
).replace('.nii.gz', '.png')

# Use the actual subject ID passed from ANAT_SURFACE_RECONSTRUCTION
actual_subject_id = '${actual_subject_id}'
published_base = Path('${params.output_dir}/fastsurfer')
fs_subject_dir_resolved = published_base / actual_subject_id

# Verify the directory exists
if not fs_subject_dir_resolved.exists():
    raise FileNotFoundError(
        f"FastSurfer subject directory not found: {fs_subject_dir_resolved}"
    )

# Generate QC
result = qc_cortical_surf_and_measures(
    fs_subject_dir=fs_subject_dir_resolved,
    output_path=Path(qc_output_filename),
    atlas_name=atlas_name,
    modality='anat',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

// ============================================
// FUNCTIONAL QC PROCESSES
// ============================================

process QC_MOTION_CORRECTION {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(motion_params_file), path(input_file), val(bids_naming_template), path(confounds_tsv)
    path config_file
    
    output:
    path "*.png", optional: true, emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_motion_correction
from nhp_mri_prep.utils.bids import create_bids_output_filename
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-motion',
    modality='bold'
).replace('.nii.gz', '.png')

def _real(p):
    # Treat dummy/empty sentinels as absent (mirrors TSNR's dummy-mask handling).
    path = Path(p)
    if (not path.exists()) or '.dummy' in str(path).lower() or path.stat().st_size == 0:
        return None
    return path

# Generate QC
result = qc_motion_correction(
    motion_params_file=Path('${motion_params_file}'),
    output_path=Path(qc_output_filename),
    input_file=Path('${input_file}'),
    config=config,
    confounds_file=_real('${confounds_tsv}')
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_BIAS_CORRECTION_FUNC {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(original_file), path(corrected_file), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_bias_correction
from nhp_mri_prep.utils.bids import create_bids_output_filename
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-biascorrect',
    modality='bold'
).replace('.nii.gz', '.png')

# Generate QC
result = qc_bias_correction(
    original_file=Path('${original_file}'),
    corrected_file=Path('${corrected_file}'),
    output_path=Path(qc_output_filename),
    modality='func',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}


process QC_CONFORM_FUNC {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(conformed_file), val(bids_naming_template), path(template_resampled_file, stageAs: 'template.nii.gz')
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_conform
from nhp_mri_prep.utils.bids import create_bids_output_filename
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Use the resampled template file from conform step (matches the conformed image space)
# File is staged as 'template.nii.gz' to avoid filename collisions
template_file = Path('template.nii.gz')

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-conform',
    modality='bold'
).replace('.nii.gz', '.png')

# Generate QC
result = qc_conform(
    conformed_file=Path('${conformed_file}'),
    template_file=template_file,
    output_path=Path(qc_output_filename),
    modality='func',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_SKULLSTRIPPING_FUNC {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(brain_file), path(mask_file), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_skullstripping
from nhp_mri_prep.utils.bids import create_bids_output_filename
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Generate BIDS-compliant QC output filename
qc_output_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-skullstrip',
    modality='bold'
).replace('.nii.gz', '.png')

# Generate QC
result = qc_skullstripping(
    underlay_file=Path('${brain_file}'),
    mask_file=Path('${mask_file}'),
    output_path=Path(qc_output_filename),
    modality='func',
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_REGISTRATION_FUNC {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(registered_file), path(reference_file), val(bids_name)
    val(modality)  // Modality for QC: 'func2anat' or 'func2target' (default)
    path config_file  // Effective config file with all resolved parameters
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_registration
from pathlib import Path
import glob
import os

# Load config
from nhp_mri_prep.utils.nextflow import load_config, detect_modality, save_metadata
from nhp_mri_prep.utils.templates import resolve_template
config = load_config('${config_file}')

# Get registered file and reference file
# Handle case where registered_file might contain multiple files (space-separated)
registered_file_str = '${registered_file}'
reference_file = Path('${reference_file}')

# If registered_file contains spaces, it means multiple files were matched
# Select the file in the final template space (not intermediate T1w space)
if ' ' in registered_file_str:
    # Split by space to get individual file paths
    file_paths = registered_file_str.split()
    # Get template name from output_space (e.g., "NMT2Sym:res-1" -> "NMT2Sym";
    # a custom template file path -> "template", matching the space-<label> outputs)
    # Get effective_output_space from effective config file
    from nhp_mri_prep.utils.nextflow import load_config
    from nhp_mri_prep.utils.templates import space_label_for
    config = load_config('${config_file}')
    effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
    template_name = space_label_for(effective_output_space)
    # Find the file in template space (final registered file)
    registered_file = None
    for fp in file_paths:
        if f'space-{template_name}' in fp:
            registered_file = Path(fp)
            break
    # Fallback: if not found, use the last file (should be template space)
    if registered_file is None or not registered_file.exists():
        registered_file = Path(file_paths[-1])
else:
    registered_file = Path(registered_file_str)

# Verify file exists
if not registered_file.exists():
    raise FileNotFoundError(f"Registered file not found: {registered_file}")

# Get modality from input (default to 'func2target' if not provided)
qc_modality = '${modality}' if '${modality}' else 'func2target'
desc_value = 'func2anat' if qc_modality == 'func2anat' else 'func2target'

# Get bids_name for filename construction
bids_name = Path('${bids_name}') if '${bids_name}' else None

# Construct QC output filename
# Keep input name and insert desc (no parse+rebuild) so custom entities (e.g. qcq) stay in order.
from nhp_mri_prep.utils.bids import get_filename_stem, create_bids_output_filename
if qc_modality == 'func2anat' and bids_name:
    qc_output_filename = create_bids_output_filename(
        original_file_path=bids_name,
        suffix=f'desc-{desc_value}',
        modality='bold'
    ).replace('.nii.gz', '.png')
else:
    # Keep registered_file name, insert _desc-{desc_value} before _bold (no parse+rebuild)
    stem = get_filename_stem(registered_file)
    if '_bold' in stem:
        stem_with_desc = stem.replace('_bold', f'_desc-{desc_value}_bold', 1)
    else:
        stem_with_desc = f"{stem}_desc-{desc_value}_bold"
    qc_output_filename = stem_with_desc + '.png'

# Generate QC
result = qc_registration(
    image_file=registered_file,
    template_file=reference_file,
    output_path=Path(qc_output_filename),
    modality=qc_modality,
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_WITHIN_SES_COREG {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'
    
    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'
    
    input:
    tuple val(subject_id), val(session_id), path(tmean_run1), path(tmean_averaged), val(bids_naming_template)
    path config_file
    
    output:
    path "*.png", emit: qc_files
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.qc import qc_within_ses_coreg
from nhp_mri_prep.utils.bids import create_bids_output_filename, parse_bids_entities, create_bids_filename
from pathlib import Path

# Load config
from nhp_mri_prep.utils.nextflow import load_config, save_metadata
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_naming_template = Path('${bids_naming_template}')

# Generate BIDS-compliant QC output filename
# Remove run-specific entities (other than sub and ses), add desc-sescoreg, ends with _boldref
# Parse the original filename to get entities
parsed = parse_bids_entities(str(bids_naming_template))
# Keep only sub and ses entities, then add desc
parsed = {k: v for k, v in parsed.items() if k in ['sub', 'ses']}
parsed['desc'] = 'sescoreg'
qc_output_filename = create_bids_filename(parsed, 'boldref', extension='.png')

# Generate QC
result = qc_within_ses_coreg(
    tmean_run1=Path('${tmean_run1}'),
    tmean_averaged=Path('${tmean_averaged}'),
    output_path=Path(qc_output_filename),
    config=config
)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process QC_TSNR {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'

    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'

    input:
    tuple val(subject_id), val(session_id), path(session_avg_nii), path(lh_surf_gii), path(rh_surf_gii), val(bids_name), val(actual_subject_id)
    path config_file

    output:
    path "*.png", optional: true, emit: qc_files
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<'PYTHON_EOF'
from pathlib import Path

from nhp_mri_prep.steps.qc import qc_tsnr
from nhp_mri_prep.utils.bids import create_bids_filename, parse_bids_entities
from nhp_mri_prep.utils.nextflow import load_config, save_metadata

# 1. Effective config and staged inputs (Nextflow copies paths into the task work dir).
config = load_config('${config_file}')
bids_naming_template = Path('${bids_name}')
session_average_tsnr_nifti = Path('${session_avg_nii}')
left_hemisphere_tsnr_gifti = Path('${lh_surf_gii}')
right_hemisphere_tsnr_gifti = Path('${rh_surf_gii}')

# 2. BIDS-compliant PNG basename from the template (sub/ses/space + desc-tSNR + bold.png).
parsed_entities = parse_bids_entities(bids_naming_template.name)
qc_filename_entities = {}
for entity_key in ('sub', 'ses', 'space'):
    if entity_key in parsed_entities:
        qc_filename_entities[entity_key] = parsed_entities[entity_key]
qc_filename_entities['desc'] = 'tSNR'
qc_output_png_filename = create_bids_filename(qc_filename_entities, 'bold', extension='.png')

# 3. FastSurfer subject directory under the published output tree (for inflated meshes).
fastsurfer_actual_subject_id = '${actual_subject_id}'.strip()
fastsurfer_output_root = Path('${params.output_dir}') / 'fastsurfer'
freesurfer_subject_directory = (
    (fastsurfer_output_root / fastsurfer_actual_subject_id)
    if fastsurfer_actual_subject_id
    else None
)
if freesurfer_subject_directory is not None and not freesurfer_subject_directory.is_dir():
    freesurfer_subject_directory = None

# 4. Dummy or empty gifti → skip surface QC (volume-only figure); qc_tsnr handles None fs dir.
if (
    '.dummy' in str(left_hemisphere_tsnr_gifti).lower()
    or '.dummy' in str(right_hemisphere_tsnr_gifti).lower()
    or not left_hemisphere_tsnr_gifti.exists()
    or not right_hemisphere_tsnr_gifti.exists()
    or left_hemisphere_tsnr_gifti.stat().st_size == 0
    or right_hemisphere_tsnr_gifti.stat().st_size == 0
):
    freesurfer_subject_directory = None

# 5. Render combined QC figure and write sidecar metadata.
result = qc_tsnr(
    session_tsnr_vol=session_average_tsnr_nifti,
    lh_surf_gii=left_hemisphere_tsnr_gifti,
    rh_surf_gii=right_hemisphere_tsnr_gifti,
    fs_subject_dir=freesurfer_subject_directory,
    output_path=Path(qc_output_png_filename),
    config=config,
)

save_metadata(result.metadata)
PYTHON_EOF
    """
}

process QC_CONFOUNDS {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    errorStrategy 'ignore'

    publishDir "${params.output_dir}/sub-${subject_id}/figures",
        mode: 'copy',
        pattern: '*.png'

    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(confounds_tsv), val(bids_name)
    path config_file

    output:
    path "*.png", optional: true, emit: qc_files
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<'PYTHON_EOF'
from pathlib import Path

from nhp_mri_prep.steps.qc import qc_confounds
from nhp_mri_prep.utils.bids import create_bids_output_filename
from nhp_mri_prep.utils.nextflow import load_config, save_metadata

config = load_config('${config_file}')
bids_naming_template = Path('${bids_name}')
confounds_tsv = Path('${confounds_tsv}')

# BIDS-compliant per-run PNG basename derived from the full bold stem, so every distinguishing
# entity (acq, dir, space, ...) is preserved -- matches QC_MOTION_CORRECTION and the other per-run
# QC processes. (An allowlist subset would collapse acq-variant runs onto one colliding filename.)
qc_output_png_filename = create_bids_output_filename(
    original_file_path=bids_naming_template,
    suffix='desc-confounds',
    modality='bold',
).replace('.nii.gz', '.png')

result = qc_confounds(
    confounds_file=confounds_tsv,
    output_path=Path(qc_output_png_filename),
    config=config,
)

save_metadata(result.metadata)
PYTHON_EOF
    """
}

// ============================================
// REPORT GENERATION
// ============================================
// QC report generation has moved out of the DAG into main.nf's
// workflow.onComplete handler (driven by nextflow_scripts/generate_reports.py),
// so a report is produced even when the run aborts early. There is intentionally
// no QC_GENERATE_REPORT process here anymore.
