/*
 * Functional processing modules for nhp_mri_prep Nextflow pipeline
 */

process FUNC_SLICE_TIMING {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(input_file), val(bids_name)
    path config_file  // Effective config file with all resolved parameters
    
    output:
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-sliceTiming_bold.nii.gz"), val(bids_name), emit: output
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.steps.functional import func_slice_timing_correction
    from nhp_mri_prep.steps.types import StepInput
    from nhp_mri_prep.utils.bids import create_bids_output_filename
    from pathlib import Path
    import shutil
    import os
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    
    # Initialize run_identifier
    run_identifier = '${run_identifier}'
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    
    # Load BIDS metadata from JSON file and update config
    from nhp_mri_prep.utils.bids import find_bids_metadata
    from nhp_mri_prep.config.bids_adapter import update_config_from_bids_metadata
    import logging
    logger = logging.getLogger(__name__)
    
    # Find BIDS dataset directory (parent of subject directory)
    bids_file_path = Path('${bids_name}')
    # Navigate up from func/ to ses/ to sub/ to dataset root
    dataset_dir = bids_file_path.parent.parent.parent.parent
    func_metadata = find_bids_metadata(bids_file_path, dataset_dir)
    
    if func_metadata:
        logger.info(f"Data: found BIDS metadata - {len(func_metadata)} keys")
        config = update_config_from_bids_metadata(config, func_metadata, logger)
        logger.info(f"Config: updated slice timing configuration from BIDS metadata")
    else:
        logger.warning(f"Data: no BIDS metadata found for {bids_file_path}")
    
    # Create step input
    input_obj = StepInput(
        input_file=Path('${input_file}'),
        working_dir=Path('work'),
        config=config,
        output_name='func_slice_timed.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
    )
    
    # Run step
    result = func_slice_timing_correction(input_obj)
    
    # Generate BIDS-compliant output filename
    bids_output_filename = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='desc-sliceTiming',
        modality='bold'
    )
    
    # Create BIDS-compliant symlink for Nextflow output and publishDir
    create_output_link(result.output_file, bids_output_filename)
    
    # Create symlinks for additional files (e.g., tmean) with BIDS-compliant names
    # Keep as symlinks until published - saves storage
    for key, f in result.additional_files.items():
        if key == 'tmean':
            # Create BIDS-compliant name for tmean (use boldref suffix)
            tmean_bids_name = create_bids_output_filename(
                original_file_path=bids_name,
                suffix='desc-sliceTiming',
                modality='boldref'
            )
            create_output_link(f, tmean_bids_name)
        else:
            create_output_link(f, f.name)
    
    # Save metadata
    save_metadata(result.metadata)
    EOF
    """
}

process FUNC_MOTION_CORRECTION {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"

    // NOTE: the canonical *_desc-confounds_timeseries.tsv is published by FUNC_COMPUTE_CONFOUNDS
    // (post-registration). Here we only stage motion params + relative-RMS as intermediate channel
    // artifacts for that downstream stage. Publishing is disabled (but a non-null path is required
    // because nextflow.config sets a pathless global publishDir default).
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false

    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(input_file), val(bids_name)
    path config_file  // Effective config file with all resolved parameters

    output:
    // Combined channel: [sub, ses, run_identifier, bold_file, tmean_file, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-motion_bold.nii.gz"), path("*desc-motion_boldref.nii.gz"), val(bids_name), emit: output
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-motion_timeseries.tsv"), emit: motion_params
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.steps.functional import func_motion_correction
    from nhp_mri_prep.steps.types import StepInput
    from nhp_mri_prep.utils.bids import create_bids_output_filename
    from pathlib import Path
    import shutil
    import os
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    
    # Initialize run_identifier
    run_identifier = '${run_identifier}'
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    
    # Create step input
    input_obj = StepInput(
        input_file=Path('${input_file}'),
        working_dir=Path('work'),
        config=config,
        output_name='func_motion_corrected.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
    )
    
    # Run step
    result = func_motion_correction(input_obj)
    
    # Generate BIDS-compliant output filename
    bids_output_filename = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='desc-motion',
        modality='bold'
    )
    
    # Create BIDS-compliant symlink for Nextflow output and publishDir
    create_output_link(result.output_file, bids_output_filename)
    
    # Create symlinks for additional files with BIDS-compliant names
    # Keep large files (tmean) as symlinks until published - saves storage
    # Small files (motion_params) are copied as they're small and may need to be actual files
    for key, f in result.additional_files.items():
        if key == 'tmean':
            # Create BIDS-compliant name for tmean (use boldref suffix)
            tmean_bids_name = create_bids_output_filename(
                original_file_path=bids_name,
                suffix='desc-motion',
                modality='boldref'
            )
            create_output_link(f, tmean_bids_name)
        elif key == 'motion_params':
            # Intermediate motion-parameter table (NOT the canonical confounds file).
            # Consumed by FUNC_COMPUTE_CONFOUNDS, which publishes desc-confounds_timeseries.tsv.
            from nhp_mri_prep.utils.bids import get_filename_stem
            original_stem = get_filename_stem(bids_name)
            bids_prefix = original_stem.replace('_bold', '')
            motion_bids_name = f"{bids_prefix}_desc-motion_timeseries.tsv"
            shutil.copy2(f, motion_bids_name)
        else:
            create_output_link(f, f.name)
    
    # Save metadata
    save_metadata(result.metadata)
    EOF
    """
}

process FUNC_GENERATE_TMEAN {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(bold_file), val(bids_name)
    path config_file
    
    output:
    // Combined channel: [sub, ses, run_identifier, bold_file, tmean_file, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("nf_out/*_bold.nii.gz"), path("*_boldref.nii.gz"), val(bids_name), emit: output
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.utils.mri import calculate_func_tmean
    from nhp_mri_prep.utils.bids import create_bids_output_filename
    from nhp_mri_prep.utils.nextflow import load_config, save_metadata, create_output_link
    from pathlib import Path
    import shutil
    import os
    import logging
    
    # Initialize logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Initialize run_identifier
    run_identifier = '${run_identifier}'
    
    # Load config
    config = load_config('${config_file}')
    
    # Get BIDS naming template
    bids_name = Path('${bids_name}')
    
    # Input BOLD file
    bold_file = Path('${bold_file}')
    
    # Write BOLD to nf_out/ subdirectory so Nextflow can collect it as an output.
    # Cannot use the staged input path directly: Nextflow excludes files from output
    # matching by their relative path in the work directory, so any file at the same
    # path as a staged input is always excluded — even if the symlink is replaced with
    # a real copy. Writing to a subdirectory gives the file a fresh relative path that
    # Nextflow has no record of, so it is collected normally. Downstream processes are
    # unaffected because Nextflow stages files by basename only.
    outdir = Path('nf_out')
    outdir.mkdir(exist_ok=True)
    bids_bold_filename = outdir / bids_name.name
    shutil.copy2(str(bold_file.resolve()), str(bids_bold_filename))
    
    # Generate tmean file
    # Convert _bold to _boldref while preserving all other parts of the filename
    tmean_basename = bids_name.name.replace('_bold.nii.gz', '_boldref.nii.gz')
    tmean_output_path = Path('work') / tmean_basename
    
    # Create work directory if it doesn't exist
    tmean_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate tmean
    logger.info(f"Step: generating temporal mean from {bold_file.name}")
    calculate_func_tmean(str(bold_file), str(tmean_output_path), logger)
    
    # Create BIDS-compliant symlink for tmean (just replace _bold with _boldref)
    bids_tmean_filename = tmean_basename
    create_output_link(tmean_output_path, bids_tmean_filename)
    
    # Save metadata
    metadata = {
        'step': 'FUNC_GENERATE_TMEAN',
        'subject_id': '${subject_id}',
        'session_id': '${session_id}' if '${session_id}' else None,
        'run_identifier': run_identifier,
        'input_file': str(bold_file),
        'output_bold': bids_bold_filename.name,
        'output_tmean': str(bids_tmean_filename)
    }
    save_metadata(metadata)
    EOF
    """
}

process FUNC_DESPIKE {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false
    
    input:
    // Combined channel: [sub, ses, run_identifier, bold_file, tmean_file, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path(bold_file), path(tmean_file), val(bids_name)
    path config_file
    
    output:
    // Combined channel: [sub, ses, run_identifier, bold_file, tmean_file, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-despike_bold.nii.gz"), path("*desc-despike_boldref.nii.gz"), val(bids_name), emit: output
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.steps.functional import func_despike
    from nhp_mri_prep.steps.types import StepInput
    from nhp_mri_prep.utils.bids import create_bids_output_filename
    from pathlib import Path
    import shutil
    import os
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    
    # Initialize run_identifier
    run_identifier = '${run_identifier}'
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    
    # Create step input (process BOLD, inherit tmean)
    input_obj = StepInput(
        input_file=Path('${bold_file}'),  # Process BOLD
        working_dir=Path('work'),
        config=config,
        output_name='func_despiked.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
    )
    
    # Run step (processes BOLD, generates new tmean)
    result = func_despike(input_obj)
    
    # Inherit tmean from input if step doesn't generate one, otherwise use generated tmean
    # func_despike generates tmean, so we use it
    
    # Generate BIDS-compliant output filename
    bids_output_filename = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='desc-despike',
        modality='bold'
    )
    
    # Create BIDS-compliant symlink for Nextflow output and publishDir
    create_output_link(result.output_file, bids_output_filename)
    
    # Create symlinks for additional files (e.g., tmean) with BIDS-compliant names
    # Keep as symlinks until published - saves storage
    for key, f in result.additional_files.items():
        if key == 'tmean' or key == 'imagef_despiked_tmean':
            # Create BIDS-compliant name for tmean (use boldref suffix)
            tmean_bids_name = create_bids_output_filename(
                original_file_path=bids_name,
                suffix='desc-despike',
                modality='boldref'
            )
            create_output_link(f, tmean_bids_name)
        else:
            create_output_link(f, f.name)
    
    # Save metadata
    save_metadata(result.metadata)
    EOF
    """
}

process FUNC_BIAS_CORRECTION {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false
    
    input:
    // Input: [sub, ses, run_identifier, tmean_file, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path(tmean_file), val(bids_name)
    path config_file
    
    output:
    // Output: [sub, ses, run_identifier, bias_corrected_tmean, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-biascorrect_boldref.nii.gz"), val(bids_name), emit: output
    path "*.json", emit: metadata
    
    script:
    """
    # Thread environment variables are set by Nextflow's beforeScript based on task.cpus
    # Python code reads OMP_NUM_THREADS from environment
    
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.functional import func_bias_correction
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import create_bids_output_filename, parse_bids_entities, create_bids_filename, get_bids_prefix
from pathlib import Path
import shutil
import os
from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata

# Load config
from nhp_mri_prep.utils.nextflow import load_config
config = load_config('${config_file}')

# Get original file path (for BIDS filename generation)
bids_name = Path('${bids_name}')
run_identifier = '${run_identifier}'

# Create step input (process tmean only)
input_obj = StepInput(
    input_file=Path('${tmean_file}'),  # Process tmean
    working_dir=Path('work'),
    config=config,
    output_name='func_bias_corrected.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
)

# Run step
result = func_bias_correction(input_obj)

# Generate BIDS-compliant output filename (bias correction operates on tmean, so use boldref)
# Use get_bids_prefix helper to determine session-level vs run-level naming
bids_prefix = get_bids_prefix(bids_name, run_identifier)
bids_output_filename = f"{bids_prefix}_desc-biascorrect_boldref.nii.gz"

# Create BIDS-compliant symlink for Nextflow output and publishDir
create_output_link(result.output_file, bids_output_filename)

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process FUNC_COMPUTE_CONFORM {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*_xfm.{mat,json}',
        saveAs: { filename -> filename == 'template_resampled.nii.gz' ? null : filename }

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*_space-bold_boldref.{nii.gz,json}'

    input:
    // Input: [sub, ses, run_identifier, tmean_file, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path(tmean_file), val(bids_name)
    path(anat_brain_file)
    path config_file  // Effective config file with all resolved parameters
    
    output:
    // Output: [sub, ses, run_identifier, conformed_tmean, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-conform_boldref.nii.gz"), val(bids_name), emit: output
    // Published BOLD reference in bold (conformed) space
    tuple val(subject_id), val(session_id), val(run_identifier), path("*_space-bold_boldref.nii.gz"), val(bids_name), emit: bold_space_reference
    // Transforms: [sub, ses, run_identifier, forward_transform, inverse_transform]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*from-scanner_to-bold_mode-image_xfm.mat"), path("*from-bold_to-scanner_mode-image_xfm.mat"), emit: transforms
    // Reference: [sub, ses, run_identifier, reference]
    tuple val(subject_id), val(session_id), val(run_identifier), path("template_resampled.nii.gz"), emit: reference
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.steps.functional import func_conform
    from nhp_mri_prep.steps.types import StepInput
    from nhp_mri_prep.utils.templates import resolve_template, space_label_for
    from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem, parse_bids_entities, create_bids_filename, get_bids_prefix
    from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
    from pathlib import Path
    import shutil
    import os
    import sys
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    run_identifier = '${run_identifier}'
    
    # Check if anatomical brain file is available
    anat_brain_path_str = '${anat_brain_file}'
    has_anat_brain = anat_brain_path_str and anat_brain_path_str.strip() != '' and '.dummy' not in anat_brain_path_str
    
    # Get effective_output_space from effective config file
    effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
    
    # Determine target file: use anatomical if available, otherwise use template
    if has_anat_brain:
        anat_brain_path = Path(anat_brain_path_str)
        if anat_brain_path.exists():
            target_file = anat_brain_path
        else:
            target_file = Path(resolve_template(effective_output_space))
    else:
        target_file = Path(resolve_template(effective_output_space))
    
    # Create step input for tmean (used for conform registration)
    tmean_input_obj = StepInput(
        input_file=Path('${tmean_file}'),
        working_dir=Path('work'),
        config=config,
        output_name='func_tmean_conformed.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
    )
    
    # Run step (only conforms tmean, does not apply to BOLD)
    result = func_conform(tmean_input_obj, target_file=target_file, bold_4d_file=None)
    
    # Generate BIDS-compliant output filename for conformed tmean
    # Use get_bids_prefix helper to determine session-level vs run-level naming
    bids_prefix = get_bids_prefix(bids_name, run_identifier)
    bids_output_filename_tmean = f"{bids_prefix}_desc-conform_boldref.nii.gz"
    
    # Create BIDS-compliant symlink for conformed tmean
    create_output_link(result.output_file, bids_output_filename_tmean)

    # Published alias: BOLD reference in bold (conformed) space (session- or run-level prefix)
    bids_bold_space_name = f"{bids_prefix}_space-bold_boldref.nii.gz"
    create_output_link(result.output_file, bids_bold_space_name)
    # Sidecar (boldref in native bold space -> no template block)
    write_derivative_sidecar(bids_bold_space_name, sources=[str(Path('${tmean_file}'))])

    # Copy transform files with BIDS-compliant names

    conform_forward_transform_path = None
    conform_inverse_transform_path = None
    for key, f in result.additional_files.items():
        if key == 'forward_transform':
            # Forward transform: from-scanner_to-bold
            bids_transform_name = f"{bids_prefix}_from-scanner_to-bold_mode-image_xfm.mat"
            shutil.copy2(f, bids_transform_name)
            conform_forward_transform_path = Path(bids_transform_name)
            # Sidecar (scanner<->bold conform transform; native, no template block)
            write_derivative_sidecar(bids_transform_name, engine=result.metadata.get('engine'), sources=[str(Path('${tmean_file}')), str(target_file)])
        elif key == 'inverse_transform':
            # Inverse transform: from-bold_to-scanner
            bids_transform_name = f"{bids_prefix}_from-bold_to-scanner_mode-image_xfm.mat"
            shutil.copy2(f, bids_transform_name)
            conform_inverse_transform_path = Path(bids_transform_name)
            write_derivative_sidecar(bids_transform_name, engine=result.metadata.get('engine'), sources=[str(Path('${tmean_file}')), str(target_file)])
        elif key == 'template_resampled':
            # Create symlink at root level for Nextflow output pattern
            # Use create_output_link() for consistency and proper symlink resolution
            reference_dest = Path('template_resampled.nii.gz')
            create_output_link(f, str(reference_dest))
        else:
            create_output_link(f, f.name)
    
    if conform_forward_transform_path is None or not conform_forward_transform_path.exists():
        raise FileNotFoundError("Forward conform transform not found or does not exist")
    
    # Inverse transform should always be created if forward exists, but handle gracefully
    if conform_inverse_transform_path is None or not conform_inverse_transform_path.exists():
        raise FileNotFoundError(
            f"Inverse conform transform not found or does not exist. "
            f"Expected: {bids_prefix}_from-bold_to-scanner_mode-image_xfm.mat. "
            f"This indicates a problem with the conform registration step."
        )
    
    # Save metadata
    save_metadata(result.metadata)
    EOF
    """
}

process FUNC_APPLY_CONFORM {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*desc-preproc*.{nii.gz,json}'

    input:
    // Input: [sub, ses, run_identifier, bold_file, conform_transform, conformed_tmean_ref, bids_template]
    // Stage conformed_tmean_ref as reg_reference.nii.gz to avoid output pattern collision
    tuple val(subject_id), val(session_id), val(run_identifier), path(bold_file), path(conform_transform), path(conformed_tmean_ref, stageAs: 'reg_reference.nii.gz'), val(bids_name)
    path config_file

    output:
    // Output: [sub, ses, run_identifier, conformed_bold, conformed_tmean_ref, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-conform_bold.nii.gz"), path("*desc-conform_boldref.nii.gz"), val(bids_name), emit: output
    // Phase 1 preproc output: [sub, ses, run_identifier, preproc_bold, preproc_boldref, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-preproc_bold.nii.gz"), path("*desc-preproc_boldref.nii.gz"), val(bids_name), emit: preproc_output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.operations.registration import flirt_apply_transforms
    from nhp_mri_prep.utils.bids import create_bids_output_filename
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    from nhp_mri_prep.utils.sidecar import write_derivative_sidecar, bold_timeseries_fields
    from pathlib import Path
    import sys
    import os
    import shutil
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    
    # Apply conform transform to BOLD
    bold_result = flirt_apply_transforms(
        movingf=str(Path('${bold_file}')),
        outputf_name='func_bold_conformed.nii.gz',
        reff=str(Path('reg_reference.nii.gz')),
        working_dir='work',
        transformf=str(Path('${conform_transform}')),
        logger=None,
        interpolation='trilinear',
        generate_tmean=False
    )
    
    if not bold_result.get("imagef_registered"):
        raise FileNotFoundError("Failed to apply conform transform to BOLD")
    
    # Generate BIDS-compliant output filename (for internal workflow use)
    bids_output_filename_bold = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='desc-conform',
        modality='bold'
    )
    
    bids_output_filename_tmean = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='desc-conform',
        modality='boldref'
    )
    
    # Ensure bids_output_filename_tmean is just a filename (not a path)
    # This is important for Nextflow pattern matching
    bids_output_filename_tmean = Path(bids_output_filename_tmean).name
    
    # Create symlinks (for internal workflow)
    create_output_link(Path(bold_result["imagef_registered"]), bids_output_filename_bold)
    
    # Generate BIDS-compliant output filename for Phase 1 preproc (published output)
    bids_preproc_filename_bold = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='space-bold_desc-preproc',
        modality='bold'
    )
    
    bids_preproc_filename_boldref = create_bids_output_filename(
        original_file_path=bids_name,
        suffix='space-bold_desc-preproc',
        modality='boldref'
    )
    
    # Ensure boldref filename is just a filename (for copying later)
    bids_preproc_filename_boldref = Path(bids_preproc_filename_boldref).name
    
    # Create symlinks for Phase 1 preproc output (published)
    bold_source = Path(bold_result["imagef_registered"])
    if not bold_source.exists():
        raise FileNotFoundError(f"Source file for preproc bold output does not exist: {bold_source}")
    
    create_output_link(bold_source, bids_preproc_filename_bold)
    
    # Verify the bold file exists and is accessible (Nextflow needs to see it)
    preproc_bold_path = Path(bids_preproc_filename_bold)
    if not preproc_bold_path.exists():
        raise FileNotFoundError(f"Failed to create preproc bold output file: {bids_preproc_filename_bold} (source: {bold_source})")
    # Check if it's a valid symlink or file
    if not (preproc_bold_path.is_symlink() or preproc_bold_path.is_file()):
        raise FileNotFoundError(f"Preproc bold output file exists but is not a valid file or symlink: {bids_preproc_filename_bold}")
    # If it's a symlink, verify it resolves correctly
    if preproc_bold_path.is_symlink():
        try:
            resolved = preproc_bold_path.resolve(strict=True)
            if not resolved.exists():
                raise FileNotFoundError(f"Preproc bold symlink points to non-existent file: {bids_preproc_filename_bold} -> {resolved}")
        except Exception as e:
            raise FileNotFoundError(f"Preproc bold symlink is broken: {bids_preproc_filename_bold}, error: {e}")
    
    # Ensure conformed_tmean_ref exists and copy it to output
    # We must COPY (not symlink) to ensure Nextflow recognizes it as a valid output
    # Nextflow excludes files that match input filenames if they're symlinks
    conformed_tmean_ref_path = Path('reg_reference.nii.gz')
    if not conformed_tmean_ref_path.exists():
        raise FileNotFoundError(f"Conformed tmean reference file not found: {conformed_tmean_ref_path}")
    
    # Ensure bids_output_filename_tmean is a Path object in the current directory
    target_path = Path(bids_output_filename_tmean)
    
    # Always copy the file (never symlink) to ensure Nextflow recognizes it as output
    # Resolve any symlinks first to get the actual file
    if conformed_tmean_ref_path.is_symlink():
        # Resolve to the actual file (follows symlink chain)
        actual_file_path = conformed_tmean_ref_path.resolve(strict=True)
        if not actual_file_path.exists():
            raise FileNotFoundError(f"Resolved conformed tmean reference file does not exist: {actual_file_path}")
        source_file = actual_file_path
    else:
        source_file = conformed_tmean_ref_path
    
    # Remove target if it exists
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    
    # Copy the file to ensure it's a distinct output file
    shutil.copy2(source_file, target_path)
    
    # Touch the file to ensure it has a new modification time
    # This helps Nextflow recognize it as a new output file
    target_path.touch()
    
    # Verify the output file was created and matches the expected pattern
    if not target_path.exists():
        raise FileNotFoundError(f"Failed to create output file: {target_path}")
    
    # Debug: Print the created filename to verify it matches the pattern *desc-conform_boldref.nii.gz
    print(f"DEBUG: Created output file: {target_path} (exists: {target_path.exists()})", file=sys.stderr)
    if not str(target_path).endswith('desc-conform_boldref.nii.gz'):
        print(f"WARNING: Output filename does not end with 'desc-conform_boldref.nii.gz': {target_path}", file=sys.stderr)
    
    # Create Phase 1 preproc boldref output (copy conformed_tmean_ref)
    target_preproc_path = Path(bids_preproc_filename_boldref)
    
    # Remove target if it exists
    if target_preproc_path.exists() or target_preproc_path.is_symlink():
        target_preproc_path.unlink()
    
    # Copy the file to ensure it's a distinct output file
    shutil.copy2(source_file, target_preproc_path)
    
    # Touch the file to ensure it has a new modification time
    target_preproc_path.touch()
    
    # Verify the output file was created
    if not target_preproc_path.exists():
        raise FileNotFoundError(f"Failed to create preproc output file: {target_preproc_path}")

    # Debug: Print the created filename to verify it matches the pattern *desc-preproc_boldref.nii.gz
    print(f"DEBUG: Created preproc output file: {target_preproc_path} (exists: {target_preproc_path.exists()})", file=sys.stderr)
    if not str(target_preproc_path).endswith('desc-preproc_boldref.nii.gz'):
        print(f"WARNING: Preproc output filename does not end with 'desc-preproc_boldref.nii.gz': {target_preproc_path}", file=sys.stderr)

    # Sidecars for the published preproc bold + boldref (native bold space -> no template block)
    _apply_conform_sources = [str(Path('${bold_file}')), str(Path('${conform_transform}'))]
    # 4D preproc bold carries TR + slice-timing-corrected flag; the 3D boldref does not.
    write_derivative_sidecar(bids_preproc_filename_bold, skull_stripped=False, sources=_apply_conform_sources, **bold_timeseries_fields(bids_name, config))
    write_derivative_sidecar(bids_preproc_filename_boldref, skull_stripped=False, sources=_apply_conform_sources)

    # Save metadata
    save_metadata({
        "step": "apply_conform",
        "modality": "func",
        "bold_file": str(Path('${bold_file}')),
        "transform_file": str(Path('${conform_transform}'))
    })
    EOF
    """
}

process FUNC_COMPUTE_BRAIN_MASK {
    label 'gpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*desc-brain_mask.{nii.gz,json}'

    input:
    // Input: [sub, ses, run_identifier, conformed_tmean, bids_template]
    tuple val(subject_id), val(session_id), val(run_identifier), path(conformed_tmean), val(bids_name)
    path config_file
    val gpu_id
    
    output:
    // Output: [sub, ses, run_identifier, masked_tmean, bids_template, brain_mask]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*_boldref_brain.nii.gz"), val(bids_name), path("*desc-brain_mask.nii.gz"), emit: output
    path "*.json", emit: metadata
    val gpu_id, emit: gpu_token
    
    script:
    """
    # Conditional GPU assignment
    if [ "${gpu_id}" != "none" ]; then
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        echo "[GPU Assignment] Task ${task.index} -> GPU ${gpu_id} (of ${params.gpu_count} available)"
    fi
    
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.steps.functional import func_skullstripping
    from nhp_mri_prep.steps.types import StepInput
    from nhp_mri_prep.utils.bids import get_filename_stem, get_bids_prefix
    from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
    from pathlib import Path
    import shutil
    import os
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    run_identifier = '${run_identifier}'
    
    # Create step input (process conformed tmean → brain)
    input_obj = StepInput(
        input_file=Path('${conformed_tmean}'),
        working_dir=Path('work'),
        config=config,
        output_name='func_brain.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
    )
    
    # Run step (processes conformed tmean to create brain)
    result = func_skullstripping(input_obj)
    
    # Generate BIDS-compliant output filename for mask
    # Use get_bids_prefix helper to determine session-level vs run-level naming
    bids_prefix_wobold = get_bids_prefix(bids_name, run_identifier)
    bids_additional_name = f"{bids_prefix_wobold}_space-bold_desc-brain_mask.nii.gz"
    
    # Create symlink for mask with BIDS-compliant name
    if "brain_mask" in result.additional_files:
        create_output_link(result.additional_files["brain_mask"], bids_additional_name)
        # Sidecar (brain mask, native bold space -> no template block)
        write_derivative_sidecar(bids_additional_name, roi_type="Brain", sources=[str(Path('${conformed_tmean}'))])
    
    # Create symlink for brain file (masked tmean)
    if result.output_file.exists():
        bids_brain_name = f"{bids_prefix_wobold}_space-bold_boldref_brain.nii.gz"
        create_output_link(result.output_file, bids_brain_name)
    
    # Save metadata
    save_metadata(result.metadata)
    EOF
    """
}

process FUNC_COMPUTE_REGISTRATION {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*.{nii.gz,h5,mat,json}',
        saveAs: { filename -> (filename.contains('ref_from_func_reg.nii.gz') || filename == 'metadata.json') ? null : filename }

    input:
    // Input: [sub, ses, run_identifier, masked_tmean, bids_template] + anatomical selection
    tuple val(subject_id), val(session_id), val(run_identifier), path(masked_tmean), val(bids_name), val(anat_session_id)
    path(anat_brain)
    path config_file  // Effective config file with all resolved parameters
    val gpu_id  // GPU ID for scheduling ('none' for CPU mode, integer for GPU mode)
    
    output:
    // Output: [sub, ses, run_identifier, registered_tmean, bids_template, anat_session_id]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*space-*boldref.nii.gz"), val(bids_name), val(anat_session_id), emit: output
    // Transforms: [sub, ses, run_identifier, forward_transform, inverse_transform]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*from-bold_to-*_mode-image_xfm.{h5,mat,nii.gz}"), path("*from-*_to-bold_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    // Reference: [sub, ses, run_identifier, reference_file]
    tuple val(subject_id), val(session_id), val(run_identifier), path("*ref_from_func_reg.nii.gz"), emit: reference
    path "*.json", emit: metadata
    val gpu_id, emit: gpu_token
    
    script:
    """
    # Conditional GPU assignment
    if [ "${gpu_id}" != "none" ]; then
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        echo "[GPU Assignment] Task ${task.index} -> GPU ${gpu_id} (of ${params.gpu_count} available)"
    fi
    
    \${PYTHON:-python3} <<EOF
    from nhp_mri_prep.steps.functional import func_registration
    from nhp_mri_prep.steps.types import StepInput
    from nhp_mri_prep.utils.templates import resolve_template, space_label_for
    from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem, parse_bids_entities, create_bids_filename, get_bids_prefix
    from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata
    from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
    from nhp_mri_prep.utils import get_image_resolution, run_command
    from pathlib import Path
    import shutil
    import os
    import sys
    import numpy as np
    
    # Load config
    from nhp_mri_prep.utils.nextflow import load_config
    config = load_config('${config_file}')
    
    # Get original file path (for BIDS filename generation)
    bids_name = Path('${bids_name}')
    run_identifier = '${run_identifier}'
    
    # Check if anatomical brain file is available
    anat_brain_path_str = '${anat_brain}'
    has_anat_brain = anat_brain_path_str and anat_brain_path_str.strip() != '' and '.dummy' not in anat_brain_path_str
    
    # Get effective_output_space from effective config file
    effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
    
    # Determine target file and type: use anatomical if available, otherwise use template
    if has_anat_brain:
        anat_brain_path = Path(anat_brain_path_str)
        if anat_brain_path.exists():
            target_file = anat_brain_path
            target_type = 'anat'
        else:
            target_file = Path(resolve_template(effective_output_space))
            target_type = 'template'
    else:
        target_file = Path(resolve_template(effective_output_space))
        target_type = 'template'
    
    # Create step input
    input_obj = StepInput(
        input_file=Path('${masked_tmean}'),
        working_dir=Path('work'),
        config=config,
        output_name='func_registered.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
    )
    
    # Run registration
    result = func_registration(input_obj, target_file=target_file, target_type=target_type)
    
    # Generate BIDS-compliant output filename
    # Determine space name based on target_type:
    # - If target_type == 'anat': space is T1w (func registered to anatomical space)
    # - If target_type == 'template': space is template_name (func registered directly to template)
    template_name = space_label_for(effective_output_space)
    if target_type == 'anat':
        space_name = "T1w"
    else:
        space_name = template_name
    
    bids_output_filename = create_bids_output_filename(
        original_file_path=bids_name,
        suffix=f'space-{space_name}_desc-preproc',
        modality='boldref'
    )
    
    # Create symlink for registered tmean
    create_output_link(result.output_file, bids_output_filename)
    # Sidecar (registered boldref; TemplateSource auto-added when target is template space)
    write_derivative_sidecar(
        bids_output_filename,
        output_space=effective_output_space,
        resolved_template_path=str(target_file),
        skull_stripped=True,
        sources=[str(Path('${masked_tmean}'))],
    )

    # Create symlinks for transform files with BIDS-compliant names (both forward and inverse)
    # .h5 files can be large, so use symlinks until published - saves storage
    # Use get_bids_prefix helper to determine session-level vs run-level naming
    bids_prefix = get_bids_prefix(bids_name, run_identifier)

    def _xfm_ext(p):
        r = Path(p).resolve()
        return ''.join(r.suffixes) if r.suffixes else r.suffix
    for key, f in result.additional_files.items():
        if key == 'forward_transform':
            ext = _xfm_ext(f)
            bids_transform_name = f"{bids_prefix}_from-bold_to-{space_name}_mode-image_xfm{ext}"
            create_output_link(f, bids_transform_name)
            write_derivative_sidecar(
                bids_transform_name,
                output_space=effective_output_space,
                resolved_template_path=str(target_file),
                engine=result.metadata.get('engine'),
                sources=[str(Path('${masked_tmean}')), str(target_file)],
            )
        elif key == 'inverse_transform':
            ext = _xfm_ext(f)
            bids_transform_name = f"{bids_prefix}_from-{space_name}_to-bold_mode-image_xfm{ext}"
            create_output_link(f, bids_transform_name)
            write_derivative_sidecar(
                bids_transform_name,
                output_space=effective_output_space,
                resolved_template_path=str(target_file),
                engine=result.metadata.get('engine'),
                sources=[str(Path('${masked_tmean}')), str(target_file)],
            )
        else:
            create_output_link(f, f.name)
    
    # Emit the original target file used during registration
    # The apply step will resample it if needed (based on keep_func_resolution)
    # Create reference file for output
    target_output = f"{bids_prefix}_ref_from_func_reg.nii.gz"
    target_path = Path(target_output)
    
    if target_file.exists():
        if target_file.resolve() != target_path.resolve():
            shutil.copy2(target_file, target_output)
            print(f"INFO: Emitted ref_from_func_reg from original target file", file=sys.stderr)
        else:
            data = target_file.read_bytes()
            target_path.write_bytes(data)
            print(f"INFO: Emitted ref_from_func_reg from original target file", file=sys.stderr)
    else:
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    # Ensure file exists
    if not target_path.exists():
        raise FileNotFoundError(f"Failed to emit ref_from_func_reg: {target_path}")
    print(f"INFO: ref_from_func_reg emitted, size: {target_path.stat().st_size} bytes", file=sys.stderr)
    
    # Save metadata
    save_metadata(result.metadata)
    EOF
    """
}

process FUNC_APPLY_TRANSFORMS {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*.{nii.gz,h5,json}',
        saveAs: { filename ->
            // Exclude intermediate QC files, internal files, and the generic metadata.json:
            if (filename.contains('target_final.nii.gz') ||
                filename.contains('_dup') ||
                filename.contains('desc-func2anat') ||
                filename == 'metadata.json') {
                return null
            }
            return filename
        }
    
    input:
    // Input structure: [sub, ses, run_identifier, bids_template, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
    // For sequential: anat2template_xfm is real file, ref_from_anat_reg is real file
    // For single: anat2template_xfm is dummy file, ref_from_anat_reg is dummy file
    // ref_from_func_reg: original target file (anat or template) from func registration, will be resampled if needed
    // target_type and target2template are now inferred from transform filename and validated against config
    tuple val(subject_id), val(session_id), val(run_identifier), val(bids_name), path(func2target_xfm), path(ref_from_func_reg), path(anat2template_xfm), path(ref_from_anat_reg)
    path(input_file)  // Input file: 4D BOLD file (conformed_bold) or 3D mask file
    val(data_type)  // "bold" or "mask" - determines interpolation, moving_type, and output handling
    path config_file  // Effective config file with all resolved parameters
    
    output:
    // For BOLD: [sub, ses, run_id, registered_bold, registered_boldref, bids_template]
    // For mask: [sub, ses, run_id, registered_mask, registered_mask (duplicate), bids_template] (duplicate mask to match structure)
    // The two slots carry distinct globs so each receives only its own suffix class.
    // Declaring one glob twice gave BOTH slots the whole sorted match list, and
    // '..._bold.nii.gz' sorts before '..._boldref.nii.gz' -- so the consumer that
    // asked for the boldref was handed the 4D BOLD, and drew its func2target QC
    // figure from it. Sequential mode still puts several files in a slot (T1w and
    // template space); picking the template-space one is the consumer's job.
    tuple val(subject_id), val(session_id), val(run_identifier), path("*space-*desc-*{_bold,_mask}.nii.gz"), path("*space-*desc-*{_boldref,_mask_dup}.nii.gz"), val(bids_name), emit: output
    // Reference file for QC: final target reference at appropriate resolution
    tuple val(subject_id), val(session_id), val(run_identifier), path("*target_final.nii.gz"), emit: reference
    // Intermediate output for sequential transforms QC: [sub, ses, run_id, func_tmean_anat, anat_reff, bids_name]
    // Only emitted when is_sequential=True, empty otherwise
    tuple val(subject_id), val(session_id), val(run_identifier), path("*space-T1w_desc-func2anat_boldref.nii.gz"), path("*_desc-func2anat_reference.nii.gz"), val(bids_name), emit: intermediate_output, optional: true
    path "*.json", emit: metadata
    
    script:
    """
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.functional import func_apply_transforms
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem, get_bids_prefix
from nhp_mri_prep.utils.nextflow import create_output_link, save_metadata, load_config
from nhp_mri_prep.utils.templates import space_label_for, resolve_template
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar, bold_timeseries_fields
from pathlib import Path
import glob
import shutil
import os
import re
import sys
import logging

# Get effective_output_space from effective config file
config = load_config('${config_file}')
effective_output_space = config.get('template', {}).get('output_space', 'NMT2Sym:res-05')
template_name = space_label_for(effective_output_space)

# Resolve the template path once for template-space sidecars (best-effort)
try:
    _resolved_template = str(resolve_template(effective_output_space)) if effective_output_space else None
except Exception:
    _resolved_template = None

# The applied moving image; sources for every produced derivative below.
_apply_input = str(Path('${input_file}'))
_apply_is_mask = '${data_type}' == 'mask'
# TR + slice-timing-corrected flag for 4D preproc bold outputs (never for masks).
_bold_fields = {} if _apply_is_mask else bold_timeseries_fields(Path('${bids_name}'), config)

def _apply_sidecar(name, *, template_space, extra_sources=None, timeseries=False):
    # Write a sidecar for a FUNC_APPLY_TRANSFORMS output. template_space=True passes the
    # template source so the TemplateSource block is populated on space-<template> files.
    # timeseries=True adds RepetitionTime/SliceTimingCorrected (4D desc-preproc_bold only).
    sources = [_apply_input] + (list(extra_sources) if extra_sources else [])
    write_derivative_sidecar(
        name,
        output_space=effective_output_space if template_space else None,
        resolved_template_path=_resolved_template if template_space else None,
        roi_type="Brain" if _apply_is_mask else None,
        sources=sources,
        **(_bold_fields if timeseries else {}),
    )

# Create logger early for use in parsing
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Config already loaded above
# Get original file path (for BIDS filename generation)
bids_name = Path('${bids_name}')
run_identifier = '${run_identifier}'

# Get input parameters
input_file = Path('${input_file}')
data_type = '${data_type}'  # "bold" or "mask"

# Set parameters based on data_type
if data_type == "mask":
    moving_type = 0  # scalar (3D mask)
    interpolation = "NearestNeighbor"  # Enforce NearestNeighbor for binary masks
    generate_tmean = False  # No temporal mean for masks
    modality = "mask"
    print(f"INFO: Processing mask with interpolation={interpolation}, moving_type={moving_type}", file=sys.stderr)
elif data_type == "bold":
    moving_type = 3  # 3D time series (4D BOLD)
    interpolation = config.get("registration", {}).get("interpolation", "BSpline")  # Use config/default
    generate_tmean = True  # Generate temporal mean for BOLD
    modality = "bold"
else:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'bold' or 'mask'")

func_4d_input = input_file  # Keep variable name for compatibility with existing code

# Get func2target transform from glob (from-bold_to-*); extension may be .h5 or .nii.gz
func2target_transform_files = [Path(f) for f in glob.glob('*from-bold_to-*_mode-image_xfm*')]
if not func2target_transform_files:
    raise FileNotFoundError("No func2target transform file found")
func2target_transform = func2target_transform_files[0]

# Parse target space from transform filename
# Pattern: from-bold_to-{space}_mode-image_xfm.{h5|nii.gz}
def extract_target_space_from_transform(transform_path: Path) -> str:
    # Extract target space from transform filename
    transform_name = transform_path.name
    # Match pattern: from-bold_to-{space}_mode-image_xfm
    match = re.search(r'from-bold_to-([^_]+)_mode-image_xfm', transform_name)
    if match:
        return match.group(1)
    else:
        # Fallback: try to extract from any pattern
        match = re.search(r'to-([^_]+)', transform_name)
        if match:
            return match.group(1)
        raise ValueError(f"Could not parse target space from transform filename: {transform_name}")

try:
    target_space = extract_target_space_from_transform(func2target_transform)
except (ValueError, AttributeError) as e:
    # Fallback: infer from ref_from_func_reg file
    # If ref_from_func_reg is not a template file, it's likely anatomical
    ref_from_func_reg_input = Path('${ref_from_func_reg}')
    has_anat_brain = not ('.dummy' in str(ref_from_func_reg_input) or 'template' in str(ref_from_func_reg_input).lower())
    if has_anat_brain:
        target_space = 'T1w'
    else:
        target_space = template_name
    logger.warning(f"Failed to parse target space from transform filename, inferred '{target_space}' from ref_from_func_reg. Error: {e}")

# Check if anatomical brain was used (inferred from ref_from_func_reg)
# If ref_from_func_reg is not a template file, it's likely anatomical
ref_from_func_reg_input = Path('${ref_from_func_reg}')
has_anat_brain = not ('.dummy' in str(ref_from_func_reg_input) or 'template' in str(ref_from_func_reg_input).lower())

# Handle anat2template_xfm - may be a single file or space-separated string
anat2template_xfm_str = '${anat2template_xfm}'
if ' ' in anat2template_xfm_str:
    # Multiple files - get the forward transform (from-T1w_to-*)
    anat2template_files = [Path(f.strip()) for f in anat2template_xfm_str.split() if f.strip()]
    anat2template_transform_path = None
    for f in anat2template_files:
        if 'from-T1w_to-' in str(f) or 'from-anat_to-' in str(f):
            anat2template_transform_path = f
            break
    if anat2template_transform_path is None:
        anat2template_transform_path = anat2template_files[0] if anat2template_files else Path('')
else:
    anat2template_transform_path = Path(anat2template_xfm_str)

ref_from_anat_reg_input = Path('${ref_from_anat_reg}')

# Check if anat2template_xfm is a dummy file
is_dummy_anat2template = '.dummy' in str(anat2template_transform_path) or not anat2template_transform_path.exists() or anat2template_transform_path == Path('')

# Check if anat_reg reference is a dummy file
is_dummy_anat_reg = '.dummy' in str(ref_from_anat_reg_input) or not ref_from_anat_reg_input.exists()

# Determine if sequential transforms are needed
# Sequential transforms: func2anat (to T1w) then anat2template (to template)
# Conditions: target space is T1w AND both anat2template transform and reference exist
is_sequential = (
    target_space == 'T1w' and 
    not is_dummy_anat2template and 
    not is_dummy_anat_reg
)

print(f"INFO: Transform application mode - target_space={target_space}, is_sequential={is_sequential}, "
      f"is_dummy_anat2template={is_dummy_anat2template}, is_dummy_anat_reg={is_dummy_anat_reg}", file=sys.stderr)

# If target space is T1w but sequential transform files are missing, warn and fallback to single transform
if target_space == 'T1w' and (is_dummy_anat2template or is_dummy_anat_reg):
    logger.warning(
        f"Target space is T1w but anat2template transform or reference is missing. "
        f"Applying single transform to T1w only. "
        f"anat2template_xfm dummy: {is_dummy_anat2template}, ref_from_anat_reg dummy: {is_dummy_anat_reg}"
    )
    is_sequential = False

# Import additional modules needed for sequential transforms
from nhp_mri_prep.operations.registration import ants_apply_transforms as ants_apply_transforms_op
from nhp_mri_prep.utils.templates import resolve_template, space_label_for
from nhp_mri_prep.utils import get_image_resolution, run_command
import numpy as np

working_dir = Path('work')
working_dir.mkdir(parents=True, exist_ok=True)

if is_sequential:
    # Sequential transforms: func2anat then anat2template
    print("INFO: Applying sequential transforms: func2anat then anat2template", file=sys.stderr)
    
    # Validate ref_from_func_reg file exists and is valid
    if not ref_from_func_reg_input.exists():
        raise FileNotFoundError(f"Ref from func reg file does not exist: {ref_from_func_reg_input}")
    file_size = ref_from_func_reg_input.stat().st_size
    if file_size == 0:
        raise ValueError(f"Ref from func reg file is empty: {ref_from_func_reg_input}")
    print(f"INFO: Ref from func reg file exists: {ref_from_func_reg_input}, size: {file_size} bytes", file=sys.stderr)
    
    # Validate anat reg reference file exists and is valid
    if is_dummy_anat_reg:
        raise FileNotFoundError(f"Anat reg reference file is missing for sequential transforms: {ref_from_anat_reg_input}")
    if not ref_from_anat_reg_input.exists():
        raise FileNotFoundError(f"Anat reg reference file does not exist: {ref_from_anat_reg_input}")
    file_size_anat = ref_from_anat_reg_input.stat().st_size
    if file_size_anat == 0:
        raise ValueError(f"Anat reg reference file is empty: {ref_from_anat_reg_input}")
    print(f"INFO: Anat reg reference file exists: {ref_from_anat_reg_input}, size: {file_size_anat} bytes", file=sys.stderr)
    
    # Step 1: Apply func2anat transform
    anat_target_name = "T1w"
    # Resample ref_from_func_reg to func resolution if needed
    if config.get("registration.keep_func_resolution", True):
        anat_reff = working_dir / "target_res-func_for_apply_transforms.nii.gz"
        func_res = np.round(get_image_resolution(str(func_4d_input), logger=logger), 1)
        cmd_resample = ['3dresample', 
                        '-input', str(ref_from_func_reg_input), '-prefix', str(anat_reff), 
                        '-rmode', 'Cu',
                        '-dxyz', str(func_res[0]), str(func_res[1]), str(func_res[2])]
        run_command(cmd_resample, step_logger=logger)
        print(f"INFO: Resampled ref_from_func_reg to func resolution for func2anat transform", file=sys.stderr)
    else:
        anat_reff = ref_from_func_reg_input
        print(f"INFO: Using ref_from_func_reg at native resolution for func2anat transform", file=sys.stderr)
    
    # Apply func2anat transform
    result_anat = ants_apply_transforms_op(
        movingf=str(func_4d_input),
        moving_type=moving_type,
        interpolation=interpolation,
        outputf_name="func2anat.nii.gz",
        fixedf=str(anat_reff),
        transformf=[str(func2target_transform)],
        reff=str(anat_reff),
        working_dir=str(working_dir),
        generate_tmean=generate_tmean,
        logger=logger
    )
    
    func_all_anat = Path(result_anat["imagef_registered"])
    func_tmean_anat = Path(result_anat.get("imagef_registered_tmean", func_all_anat))
    
    # Save data in anat space (intermediate output, not published for masks)
    # For masks, use desc-brain; for BOLD, use desc-preproc
    if data_type == "mask":
        # Remove _bold from bids_name before creating mask filename
        bids_template_for_mask = Path(str(bids_name).replace('_bold', ''))
        func_anat_output_name = create_bids_output_filename(
            original_file_path=bids_template_for_mask,
            suffix=f'space-{anat_target_name}_desc-brain',
            modality='mask'
        )
    else:
        func_anat_output_name = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{anat_target_name}_desc-preproc',
            modality=modality
        )
    create_output_link(func_all_anat, func_anat_output_name)
    _apply_sidecar(func_anat_output_name, template_space=False, extra_sources=[str(func2target_transform)], timeseries=True)

    # Save boldref in anat space (only for BOLD)
    if data_type == "bold":
        func_anat_boldref_name = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{anat_target_name}_desc-preproc',
            modality='boldref'
        )
        create_output_link(func_tmean_anat, func_anat_boldref_name)
        _apply_sidecar(func_anat_boldref_name, template_space=False, extra_sources=[str(func2target_transform)])
    
    # Create intermediate output files for QC (func2anat step)
    # These are only needed for QC snapshots, not for the pipeline
    if data_type == "bold":
        # Create intermediate boldref file for QC (BIDS-compliant naming)
        intermediate_boldref_name = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{anat_target_name}_desc-func2anat',
            modality='boldref'
        )
        create_output_link(func_tmean_anat, intermediate_boldref_name)
        
        # Create intermediate reference file for QC (anat_reff used in func2anat transform)
        # Use get_bids_prefix helper for session-level vs run-level naming
        bids_prefix = get_bids_prefix(bids_name, run_identifier)
        intermediate_reference_name = f"{bids_prefix}_desc-func2anat_reference.nii.gz"
        create_output_link(anat_reff, intermediate_reference_name)
        print(f"INFO: Created intermediate files for func2anat QC: {intermediate_boldref_name}, {intermediate_reference_name}", file=sys.stderr)
    
    # Step 2: Apply anat2template transform
    # Use ref_from_anat_reg (template file from anat registration)
    template_fixedf = ref_from_anat_reg_input
    print(f"INFO: Using ref_from_anat_reg (template) for anat2template transform", file=sys.stderr)
    
    # Resample template to func resolution if needed
    if config.get("registration.keep_func_resolution", True):
        template_reff = working_dir / "template_res-func_for_apply_transforms.nii.gz"
        func_res = np.round(get_image_resolution(str(func_all_anat), logger=logger), 1)
        cmd_resample = ['3dresample', 
                        '-input', str(template_fixedf), '-prefix', str(template_reff), 
                        '-rmode', 'Cu',
                        '-dxyz', str(func_res[0]), str(func_res[1]), str(func_res[2])]
        run_command(cmd_resample, step_logger=logger)
        print(f"INFO: Template resampled to func resolution", file=sys.stderr)
    else:
        template_reff = template_fixedf
    
    # Apply anat2template transform
    result_template = ants_apply_transforms_op(
        movingf=str(func_all_anat),
        moving_type=moving_type,
        interpolation=interpolation,
        outputf_name="func2template.nii.gz",
        fixedf=str(template_fixedf),
        transformf=[str(anat2template_transform_path)],
        reff=str(template_reff),
        working_dir=str(working_dir),
        generate_tmean=generate_tmean,
        logger=logger
    )
    
    func_all_template = Path(result_template["imagef_registered"])
    func_tmean_template = Path(result_template.get("imagef_registered_tmean", func_all_template))
    
    # Save data in template space (final output)
    if data_type == "mask":
        # For mask, use desc-brain instead of desc-preproc
        # Remove _bold from bids_name before creating mask filename
        # (similar to how boldref handles it)
        bids_template_for_mask = Path(str(bids_name).replace('_bold', ''))
        func_template_output_name = create_bids_output_filename(
            original_file_path=bids_template_for_mask,
            suffix=f'space-{template_name}_desc-brain',
            modality='mask'
        )
    else:
        func_template_output_name = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{template_name}_desc-preproc',
            modality='bold'
        )
    create_output_link(func_all_template, func_template_output_name)
    _apply_sidecar(func_template_output_name, template_space=True, extra_sources=[str(func2target_transform), str(anat2template_transform_path)], timeseries=True)

    # Save boldref in template space (final output) - only for BOLD
    # For mask, create a duplicate symlink to match BOLD output structure
    if data_type == "bold":
        func_template_boldref_name = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{template_name}_desc-preproc',
            modality='boldref'
        )
        create_output_link(func_tmean_template, func_template_boldref_name)
        _apply_sidecar(func_template_boldref_name, template_space=True, extra_sources=[str(func2target_transform), str(anat2template_transform_path)])
    else:
        # For mask, create a second symlink (duplicate) to match BOLD output structure [bold, boldref]
        mask_second_name = str(func_template_output_name).replace('_mask.nii.gz', '_mask_dup.nii.gz')
        create_output_link(func_all_template, mask_second_name)
    
    # Output final reference file for QC: template at appropriate resolution
    # Sequential transforms: final space is template
    # Reuse template_reff (already resampled if needed) instead of resampling again
    # Use get_bids_prefix helper for session-level vs run-level naming
    bids_prefix = get_bids_prefix(bids_name, run_identifier)
    target_final_output = f"{bids_prefix}_target_final.nii.gz"
    target_final_path = Path(target_final_output)
    
    # Use template_reff (already resampled to func resolution if keep_func_resolution=True)
    # This avoids duplicate resampling - template_reff was already created above
    # Use symlink - keep as symlink until published - saves storage
    if template_reff.exists():
        create_output_link(template_reff, target_final_output)
        print(f"INFO: Created target_final.nii.gz from template_reff (reused from transform)", file=sys.stderr)
    else:
        raise FileNotFoundError(f"Template reference file not found: {template_reff}")
    
    # Ensure file exists
    if not target_final_path.exists():
        raise FileNotFoundError(f"Failed to create target_final.nii.gz: {target_final_path}")
    print(f"INFO: target_final.nii.gz created, size: {target_final_path.stat().st_size} bytes", file=sys.stderr)
    
    # Final outputs are in template space
    final_output_file = func_all_template
    if data_type == "bold":
        final_boldref_file = func_tmean_template
    else:
        # For mask, duplicate the mask file to match BOLD output structure [bold, boldref]
        final_boldref_file = func_all_template
    final_space_name = template_name
    
else:
    # Single transform: func2anat or func2template
    print(f"INFO: Applying single transform: func2{target_space}", file=sys.stderr)
    
    # Use parsed target space for BIDS naming
    if target_space == 'T1w':
        target_name = "T1w"
    else:
        target_name = target_space  # Should be template name
    
    # Prepare reference file - resample ref_from_func_reg to func resolution if needed
    if config.get("registration.keep_func_resolution", True):
        reff = working_dir / "target_res-func_for_apply_transforms.nii.gz"
        func_res = np.round(get_image_resolution(str(func_4d_input), logger=logger), 1)
        cmd_resample = ['3dresample', 
                        '-input', str(ref_from_func_reg_input), '-prefix', str(reff), 
                        '-rmode', 'Cu',
                        '-dxyz', str(func_res[0]), str(func_res[1]), str(func_res[2])]
        run_command(cmd_resample, step_logger=logger)
        print(f"INFO: Resampled ref_from_func_reg to func resolution for transform", file=sys.stderr)
    else:
        reff = ref_from_func_reg_input
        print(f"INFO: Using ref_from_func_reg at native resolution for transform", file=sys.stderr)
    
    # Validate reference file exists and is valid
    if not reff.exists():
        raise FileNotFoundError(f"Reference file does not exist: {reff}")
    file_size = reff.stat().st_size
    if file_size == 0:
        raise ValueError(f"Reference file is empty: {reff}")
    print(f"INFO: Reference file exists: {reff}, size: {file_size} bytes", file=sys.stderr)
    
    # Apply single transform
    # Pass interpolation parameter explicitly to ensure masks use NearestNeighbor
    result = func_apply_transforms(
        StepInput(
            input_file=func_4d_input,
            working_dir=working_dir,
            config=config,
            output_name='func_registered.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
        ),
        transform_files=[func2target_transform],
        reference_file=reff,
        interpolation=interpolation  # Explicitly pass interpolation (NearestNeighbor for masks, config for BOLD)
    )
    
    # Generate BIDS-compliant output filename with space entity
    if data_type == "mask":
        # Remove _bold from bids_name before creating mask filename
        bids_template_for_mask = Path(str(bids_name).replace('_bold', ''))
        bids_output_filename = create_bids_output_filename(
            original_file_path=bids_template_for_mask,
            suffix=f'space-{target_name}_desc-brain',
            modality='mask'
        )
    else:
        bids_output_filename = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{target_name}_desc-preproc',
            modality='bold'
        )
    create_output_link(result.output_file, bids_output_filename)
    # Sidecar; TemplateSource auto-added only when space-<template> (target_name != T1w)
    _apply_sidecar(bids_output_filename, template_space=True, extra_sources=[str(func2target_transform)], timeseries=True)

    # Generate BIDS-compliant output filename for tmean (boldref) - only for BOLD
    if data_type == "bold" and "tmean" in result.additional_files:
        bids_boldref_filename = create_bids_output_filename(
            original_file_path=bids_name,
            suffix=f'space-{target_name}_desc-preproc',
            modality='boldref'
        )
        create_output_link(result.additional_files["tmean"], bids_boldref_filename)
        _apply_sidecar(bids_boldref_filename, template_space=True, extra_sources=[str(func2target_transform)])
    elif data_type == "mask":
        # For mask, create a second symlink (duplicate) to match BOLD output structure [bold, boldref]
        mask_second_name = str(bids_output_filename).replace('_mask.nii.gz', '_mask_dup.nii.gz')
        create_output_link(result.output_file, mask_second_name)
    
    # Output final reference file for QC: target at appropriate resolution
    # Use the same reference file that was used for the transform (already resampled if needed)
    # Use get_bids_prefix helper for session-level vs run-level naming
    bids_prefix = get_bids_prefix(bids_name, run_identifier)
    target_final_output = f"{bids_prefix}_target_final.nii.gz"
    target_final_path = Path(target_final_output)
    
    # Use the same reference file that was used for the transform (already resampled if needed)
    # Use symlink - keep as symlink until published - saves storage
    if reff.exists():
        create_output_link(reff, target_final_output)
        print(f"INFO: Created target_final.nii.gz from reference (same as used for transform)", file=sys.stderr)
    else:
        raise FileNotFoundError(f"Reference file not found for target_final.nii.gz: {reff}")
    
    # Ensure file exists
    if not target_final_path.exists():
        raise FileNotFoundError(f"Failed to create target_final.nii.gz: {target_final_path}")
    print(f"INFO: target_final.nii.gz created, size: {target_final_path.stat().st_size} bytes", file=sys.stderr)
    
    final_output_file = result.output_file
    if data_type == "bold":
        final_boldref_file = result.additional_files.get("tmean", result.output_file)
    else:
        # For mask, duplicate the mask file to match BOLD output structure [bold, boldref]
        final_boldref_file = result.output_file
    final_space_name = target_name

# Save metadata
# Determine target_type for metadata (inferred from target_space)
metadata_target_type = 'anat' if target_space == 'T1w' else 'template'
metadata_target2template = is_sequential  # True if sequential transforms were applied

save_metadata({
    "step": "apply_transforms",
    "modality": "func" if data_type == "bold" else "mask",
    "data_type": data_type,
    "target_type": metadata_target_type,
    "target2template": metadata_target2template,
    "target_space": target_space,  # Add parsed space for reference
    "is_sequential": is_sequential,
    "space": final_space_name,
    "interpolation": interpolation,
    "moving_type": moving_type
})

EOF
    """
}

/*
 * Additional functional processes for within-session coregistration
 */

process FUNC_WITHIN_SES_COREG {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false
    
    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(bold_file), path(tmean_file), val(bids_name)
    path(reference_tmean)
    val(reference_run_identifier)
    path config_file
    val gpu_id  // GPU ID for scheduling ('none' for CPU mode, integer for GPU mode)
    
    output:
    tuple val(subject_id), val(session_id), val(run_identifier), path("*desc-coreg_bold.nii.gz"), path("*desc-coreg_boldref.nii.gz"), val(bids_name), emit: output
    tuple val(subject_id), val(session_id), val(run_identifier), path("from-${run_identifier}_to-${reference_run_identifier}_mode-image_xfm.{h5,mat,nii.gz}"), emit: transforms
    path "*.json", emit: metadata
    val gpu_id, emit: gpu_token
    
    script:
    """
    # Conditional GPU assignment (ants_register uses FireANTs when GPU available)
    if [ "${gpu_id}" != "none" ]; then
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        echo "[GPU Assignment] Task ${task.index} -> GPU ${gpu_id} (of ${params.gpu_count} available)"
    fi
    
    \${PYTHON:-python3} <<EOF
from nhp_mri_prep.steps.functional import func_within_ses_coreg
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import create_bids_output_filename, get_filename_stem
from nhp_mri_prep.utils.nextflow import (
    load_config, detect_modality, save_metadata, create_output_link
)
from pathlib import Path
import shutil
import os

# Initialize run_identifier
run_identifier = '${run_identifier}'

# Load config
config = load_config('${config_file}')

# Get BIDS naming template (for BIDS filename generation)
bids_name = Path('${bids_name}')

# Get reference tmean and run_identifier
reference_tmean = Path('${reference_tmean}')
reference_run_identifier = '${reference_run_identifier}'

# Get BOLD file
bold_file = Path('${bold_file}')

# Create step input (tmean is the input file for coregistration)
input_obj = StepInput(
    input_file=Path('${tmean_file}'),
    working_dir=Path('work'),
    config=config,
    output_name='tmean_coregistered.nii.gz',
        metadata={
            'subject_id': '${subject_id}',
            'session_id': '${session_id}',
            'run_identifier': run_identifier
        }
)

# Run step
result = func_within_ses_coreg(
    input_obj,
    reference_tmean=reference_tmean,
    reference_run=reference_run_identifier,
    current_run=run_identifier,
    bold_file=bold_file
)

# Generate BIDS-compliant output filename for tmean (with desc-coreg)
bids_tmean_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-coreg',
    modality='boldref'
)

# Generate BIDS-compliant output filename for BOLD (with desc-coreg)
bids_bold_filename = create_bids_output_filename(
    original_file_path=bids_name,
    suffix='desc-coreg',
    modality='bold'
)

# Create symlinks for coregistered BOLD and tmean (these match the output pattern)
create_output_link(result.output_file, bids_tmean_filename)
if "bold_coregistered" in result.additional_files:
    create_output_link(result.additional_files["bold_coregistered"], bids_bold_filename)
else:
    raise FileNotFoundError("Coregistered BOLD file not found in result")

# Generate BIDS prefix (filename stem without modality)
original_stem = get_filename_stem(bids_name)
bids_prefix_wo_modality = original_stem.replace("_bold", "").replace("_boldref", "")

# Create symlinks for transform files (keep nature suffix: .nii.gz or .h5) for Nextflow pattern matching
def _xfm_ext(p):
    r = Path(p).resolve()
    return ''.join(r.suffixes) if r.suffixes else r.suffix
if "forward_transform" in result.additional_files:
    ext = _xfm_ext(result.additional_files["forward_transform"])
    create_output_link(result.additional_files["forward_transform"], f"from-${run_identifier}_to-${reference_run_identifier}_mode-image_xfm{ext}")

if "inverse_transform" in result.additional_files:
    # Create the expected output file name for Nextflow pattern matching (if needed)
    # Note: inverse transform is not currently used in output, but create it for completeness
    pass  # Inverse transform not needed for output pattern matching

# Save metadata
save_metadata(result.metadata)
EOF
    """
}

process FUNC_AVERAGE_TMEAN {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*desc-coreg_boldref.{nii.gz,json}',
        saveAs: { filename -> filename }

    input:
    val(tmean_files)  // List of tmean file paths (as strings) - direct input, avoids staging name collisions and file list overhead
    val(subject_id)
    val(session_id)
    val(bids_name)
    path config_file
    
    output:
    tuple val(subject_id), val(session_id), path("*desc-coreg_boldref.nii.gz"), val(bids_name), emit: output
    path "*.json", emit: metadata
    
    script:
    """
    
    \${PYTHON:-python3} <<'PYTHON_EOF'
from nhp_mri_prep.steps.functional import func_average_tmean
from nhp_mri_prep.utils.bids import create_bids_output_filename, parse_bids_entities, create_bids_filename
from nhp_mri_prep.utils.nextflow import (
    load_config, save_metadata, create_output_link
)
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from pathlib import Path
import shutil
import os
import json
import sys
import ast

# Load config
config = load_config('${config_file}')

# Get BIDS naming template
bids_name = Path('${bids_name}')

# Parse tmean_files from Nextflow (passed as JSON string for reliable parsing)
tmean_files_json = '${tmean_files}'
tmean_files_list = []

# Parse JSON (most reliable method)
try:
    parsed = json.loads(tmean_files_json)
    if isinstance(parsed, list):
        tmean_files_list = parsed
    else:
        # Single file wrapped in JSON
        tmean_files_list = [parsed]
except json.JSONDecodeError:
    # Fallback: If JSON parsing fails, try ast.literal_eval (for backward compatibility)
    try:
        parsed = ast.literal_eval(tmean_files_json)
        if isinstance(parsed, list):
            tmean_files_list = parsed
        else:
            tmean_files_list = [parsed]
    except:
        # Last resort: treat as single file path
        cleaned = tmean_files_json.strip().strip('"').strip("'")
        if cleaned:
            tmean_files_list = [cleaned]

# Convert to Path objects and filter to only existing files
tmean_file_paths = []
for file_path_str in tmean_files_list:
    if not file_path_str:
        continue
    # Final cleanup - remove any remaining brackets or quotes
    file_path_str = file_path_str.strip().strip('[').strip(']').strip('"').strip("'")
    if not file_path_str:
        continue
    file_path = Path(file_path_str)
    # Resolve to absolute path if relative
    if not file_path.is_absolute():
        file_path = file_path.resolve()
    if file_path.exists():
        tmean_file_paths.append(file_path)
    else:
        print(f"Warning: Tmean file does not exist: {file_path}", file=sys.stderr)

if len(tmean_file_paths) == 0:
    raise ValueError(f"No valid tmean files found for averaging. Input was: {tmean_files_json}")

# Create working directory if it doesn't exist
work_dir = Path('work')
work_dir.mkdir(parents=True, exist_ok=True)

# Run averaging
result = func_average_tmean(
    tmean_files=tmean_file_paths,
    working_dir=work_dir,
    config=config
)

# Generate BIDS-compliant output filename
# Remove all run-specific entities (keep only sub, ses), add desc-coreg, ends with _boldref
# Format: sub-XX[_ses-XX]_desc-coreg_boldref.nii.gz
# Parse the original filename to get entities
parsed = parse_bids_entities(str(bids_name))
# Keep only sub and ses entities (remove all run-specific identifiers like task, run, acq, etc.)
# Create a new dict with only sub and ses
filtered_entities = {}
if 'sub' in parsed:
    filtered_entities['sub'] = parsed['sub']
if 'ses' in parsed:
    filtered_entities['ses'] = parsed['ses']
# Add desc entity for coregistration
filtered_entities['desc'] = 'coreg'
filtered_entities['space'] = 'scanner'
# Rebuild filename with suffix 'boldref'
bids_output_filename = create_bids_filename(filtered_entities, 'boldref', extension='.nii.gz')

# Create symlink with BIDS-compliant name (this matches the output pattern)
create_output_link(result.output_file, bids_output_filename)
# Sidecar (session-averaged coreg boldref, native scanner space -> no template block)
write_derivative_sidecar(bids_output_filename, sources=[str(p) for p in tmean_file_paths])

# Save metadata
save_metadata(result.metadata)
PYTHON_EOF
    """
}

process FUNC_COMPUTE_CONFOUNDS {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"
    // Confounds are an additive derivative; a failure on one run should not abort the dataset run.
    errorStrategy 'ignore'

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*_desc-confounds_timeseries.{tsv,json}'

    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(bold_4d), path(mask_file), path(motion_file), val(bids_name), path(seg_file), path(seg_lut_file)
    path config_file

    output:
    // Optional: a degenerate run (e.g. a single-volume BOLD with no timeseries) is skipped as a
    // clean success that emits nothing, rather than riding the errorStrategy 'ignore' failure path.
    tuple val(subject_id), val(session_id), val(run_identifier), path("*_desc-confounds_timeseries.tsv"), val(bids_name), optional: true, emit: output
    path "*_desc-confounds_timeseries.json", optional: true, emit: sidecar

    script:
    """
    \${PYTHON:-python3} <<'PYTHON_EOF'
import sys
from pathlib import Path
import shutil

import nibabel as nib

from nhp_mri_prep.steps.functional import func_compute_confounds
from nhp_mri_prep.steps.types import StepInput
from nhp_mri_prep.utils.bids import get_filename_stem
from nhp_mri_prep.utils.nextflow import load_config

run_identifier = '${run_identifier}'
config = load_config('${config_file}')
bids_name = Path('${bids_name}')

# Confounds require a timeseries. A degenerate run (not 4D, or < 2 volumes) has no FD/DVARS/
# derivative columns to compute, so skip it as a clean success (no output) instead of erroring.
_bold_shape = nib.load('${bold_4d}').shape
if len(_bold_shape) < 4 or _bold_shape[3] < 2:
    print(f"INFO: skipping confounds - single-volume / non-timeseries run (shape {_bold_shape})")
    sys.exit(0)

def _real(p):
    # Treat dummy/empty sentinels as absent (mirrors TSNR's dummy-mask handling).
    path = Path(p)
    if (not path.exists()) or '.dummy' in str(path).lower() or path.stat().st_size == 0:
        return None
    return path

input_obj = StepInput(
    input_file=Path('${bold_4d}'),
    working_dir=Path('work'),
    config=config,
    output_name='confounds',
    metadata={
        'subject_id': '${subject_id}',
        'session_id': '${session_id}',
        'run_identifier': run_identifier,
    },
)

result = func_compute_confounds(
    input_obj,
    motion_par_file=_real('${motion_file}'),
    brain_mask_file=_real('${mask_file}'),
    seg_file=_real('${seg_file}'),          # T1w-space segmentation (None -> no tissue regressors)
    seg_lut_file=_real('${seg_lut_file}'),  # atlas LUT for tissue classification
)

# Publish canonical fMRIPrep-style names: <prefix>_desc-confounds_timeseries.{tsv,json}
bids_prefix = get_filename_stem(bids_name).replace('_bold', '')
tsv_out = f"{bids_prefix}_desc-confounds_timeseries.tsv"
json_out = f"{bids_prefix}_desc-confounds_timeseries.json"
shutil.copy2(result.output_file, tsv_out)
if result.additional_files.get('confounds_json'):
    shutil.copy2(result.additional_files['confounds_json'], json_out)
PYTHON_EOF
    """
}

process FUNC_COMPUTE_TSNR {
    label 'cpu'
    tag "${subject_id}_${session_id}_${run_identifier}"

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*_stat-tsnr_boldmap.{nii.gz,json}'

    input:
    tuple val(subject_id), val(session_id), val(run_identifier), path(bold_4d), val(bids_name), path(mask_file), val(bold_space)
    path config_file

    output:
    tuple val(subject_id), val(session_id), val(run_identifier), path('*_stat-tsnr_boldmap.nii.gz*'), val(bids_name), val(bold_space), emit: output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<'PYTHON_EOF'
from pathlib import Path

from nhp_mri_prep.steps.tsnr import compute_tsnr_run
from nhp_mri_prep.utils.nextflow import save_metadata, load_config
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from nhp_mri_prep.utils.templates import resolve_template

bids_name = Path('${bids_name}')
run_identifier = '${run_identifier}'
bold_4d = Path('${bold_4d}')
mask = Path('${mask_file}')

# 1. Drop dummy / empty mask so compute_tsnr_run ignores it like a missing mask.
if not mask.exists() or '.dummy' in str(mask).lower() or mask.stat().st_size == 0:
    mask = None

# 2. Output basename mirrors the staged BOLD: swap the BOLD modality tail for the tSNR map suffix.
out_name = bold_4d.name.replace('_bold.nii.gz', '_stat-tsnr_boldmap.nii.gz')

result = compute_tsnr_run(
    bold_4d,
    Path(out_name),
    mask_file=mask,
    min_n_tp=10,
    bold_space='${bold_space}',
)

meta = dict(result.metadata)
meta['subject_id'] = '${subject_id}'
meta['session_id'] = '${session_id}'
meta['run_identifier'] = run_identifier
meta['bids_name'] = str(bids_name)
meta['bold_space'] = '${bold_space}'
save_metadata(meta)

# Sidecar for the tSNR map (TemplateSource auto-added when the map is in template space).
if not meta.get('skipped') and Path(out_name).exists():
    _tsnr_output_space = load_config('${config_file}').get('template', {}).get('output_space')
    try:
        _tsnr_template = str(resolve_template(_tsnr_output_space)) if _tsnr_output_space else None
    except Exception:
        _tsnr_template = None
    write_derivative_sidecar(
        Path(out_name),
        output_space=_tsnr_output_space,
        resolved_template_path=_tsnr_template,
        sources=[str(bold_4d)] + ([str(mask)] if mask else []),
    )

if meta.get('skipped'):
    # Nextflow output contract: a skipped run produces no tSNR map, so emit a .dummy
    # sentinel that satisfies the process output and is filtered out downstream.
    Path(out_name + '.dummy').write_text('')
    print('INFO: tSNR run skipped:', meta.get('reason', meta.get('error', '')), file=__import__('sys').stderr)
PYTHON_EOF
    """
}

process FUNC_TSNR_SESSION_AVERAGE {
    label 'cpu'
    tag "${subject_id}_${session_id}"

    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        pattern: '*_stat-tsnr_boldmap.{nii.gz,json}'

    input:
    tuple val(subject_id), val(session_id), val(run_identifier), val(tsnr_paths_json), val(bids_name), val(bold_space)
    path config_file

    output:
    tuple val(subject_id), val(session_id), path('*_stat-tsnr_boldmap.nii.gz'), val(bids_name), val(bold_space), emit: output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<'PYTHON_EOF'
from pathlib import Path

import json
import re

from nhp_mri_prep.steps.tsnr import compute_tsnr_session_avg
from nhp_mri_prep.utils.nextflow import save_metadata, load_config
from nhp_mri_prep.utils.sidecar import write_derivative_sidecar
from nhp_mri_prep.utils.templates import resolve_template

bids_name = Path('${bids_name}')
run_identifier = '${run_identifier}'
tsnr_paths_json = '${tsnr_paths_json}'

# 1. Parse the JSON list of run-level tSNR paths produced upstream (groupTuple / JsonOutput).
paths_from_json = json.loads(tsnr_paths_json)
if not isinstance(paths_from_json, list):
    paths_from_json = [paths_from_json]

# 2. Pick the first path that exists in this task work directory (ordering matches the session's runs).
first_existing_run_tsnr_path = None
for path_entry in paths_from_json:
    if not path_entry:
        continue
    candidate_path = Path(str(path_entry).strip().strip('"').strip("'"))
    if candidate_path.exists():
        first_existing_run_tsnr_path = candidate_path
        break
if first_existing_run_tsnr_path is None:
    raise ValueError(f'No valid run-level tSNR path in JSON: {tsnr_paths_json}')

# 3. Filename stem without extension (prefer explicit .nii.gz strip for a single predictable rule).
run_tsnr_filename = first_existing_run_tsnr_path.name
if run_tsnr_filename.endswith('.nii.gz'):
    session_avg_stem = run_tsnr_filename[: -len('.nii.gz')]
else:
    session_avg_stem = first_existing_run_tsnr_path.stem

# 4. Parse run_identifier entities and remove each "_key-value" token from the stem.
#    This avoids fragile full-substring matching when entity order differs.
run_key_fragment = (run_identifier or '').strip()
if run_key_fragment:
    entity_keys = re.findall(r'([a-zA-Z]+)-[a-zA-Z0-9-]+', run_key_fragment)
    for key in entity_keys:
        session_avg_stem = re.sub(rf'_{re.escape(key)}-[^_]+', '', session_avg_stem)

# 5. Write the averaged map under the session-level BIDS basename.
out_name = session_avg_stem + '.nii.gz'

result = compute_tsnr_session_avg(tsnr_paths_json, Path(out_name))

meta = dict(result.metadata)
meta['subject_id'] = '${subject_id}'
meta['session_id'] = '${session_id}'
meta['run_identifier'] = run_identifier
meta['bids_name'] = str(bids_name)
meta['bold_space'] = '${bold_space}'
save_metadata(meta)

# Sidecar for the session-average tSNR map (TemplateSource auto-added in template space).
if Path(out_name).exists():
    _tsnr_output_space = load_config('${config_file}').get('template', {}).get('output_space')
    try:
        _tsnr_template = str(resolve_template(_tsnr_output_space)) if _tsnr_output_space else None
    except Exception:
        _tsnr_template = None
    write_derivative_sidecar(
        Path(out_name),
        output_space=_tsnr_output_space,
        resolved_template_path=_tsnr_template,
        sources=[str(p) for p in paths_from_json if p],
    )
PYTHON_EOF
    """
}

process FUNC_TSNR_SURF_PROJECTION {
    label 'cpu'
    tag "${subject_id}_${session_id}"
    errorStrategy 'ignore'

    // Downstream QC reads outputs from the work dir only; no need to copy into output_dir.
    // Global process.publishDir is a map without `path`; a process with no publishDir can
    // merge to a null target on finalize. Match FUNC_SLICE_TIMING: explicit path + enabled: false.
    publishDir "${params.output_dir}/sub-${subject_id}${session_id ? "/ses-${session_id}" : ""}/func",
        mode: 'copy',
        enabled: false

    input:
    tuple val(subject_id), val(session_id), path(session_avg_nii), val(bids_name), val(actual_subject_id)
    path config_file

    output:
    tuple val(subject_id), val(session_id), path('lh_tsnr_projection.surf.gii'), path('rh_tsnr_projection.surf.gii'), val(bids_name), emit: output
    path "*.json", emit: metadata

    script:
    """
    \${PYTHON:-python3} <<'PYTHON_EOF'
from pathlib import Path
import shutil

from nhp_mri_prep.steps.tsnr import project_tsnr_to_surface
from nhp_mri_prep.utils.nextflow import save_metadata

session_avg = Path('${session_avg_nii}')
bids_name = Path('${bids_name}')
actual_id = '${actual_subject_id}'.strip()

published_base = Path('${params.output_dir}') / 'fastsurfer'
fs_subject_dir = (published_base / actual_id) if actual_id else None
if fs_subject_dir is not None and not fs_subject_dir.is_dir():
    fs_subject_dir = None

meta = {
    'step': 'func_tsnr_surf_projection',
    'subject_id': '${subject_id}',
    'session_id': '${session_id}',
    'bids_name': str(bids_name),
    'actual_subject_id': actual_id,
    'fs_subject_dir': str(fs_subject_dir) if fs_subject_dir else None,
}

lh_out = Path('lh_tsnr_projection.surf.gii')
rh_out = Path('rh_tsnr_projection.surf.gii')

if fs_subject_dir is None:
    meta['skipped'] = True
    meta['reason'] = 'no_fastsurfer_subject_dir'
    lh_out.write_bytes(b'')
    rh_out.write_bytes(b'')
else:
    res = project_tsnr_to_surface(session_avg, fs_subject_dir, Path('.'))
    meta.update({k: str(v) if isinstance(v, Path) else v for k, v in res.items() if k in ('skipped', 'reason')})
    lh_gii = res.get('lh_gii')
    rh_gii = res.get('rh_gii')
    if lh_gii and rh_gii and Path(lh_gii).is_file() and Path(rh_gii).is_file():
        shutil.copy2(lh_gii, lh_out)
        shutil.copy2(rh_gii, rh_out)
    else:
        meta['skipped'] = True
        lh_out.write_bytes(b'')
        rh_out.write_bytes(b'')

save_metadata(meta)
PYTHON_EOF
    """
}
