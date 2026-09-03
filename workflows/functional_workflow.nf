/*
 * Functional Processing Workflow
 * 
 * Handles all functional processing steps including:
 * - Functional job parsing
 * - Functional processing pipeline (slice timing, motion correction, despike, bias correction, conform, skull stripping, registration)
 * - Within-session coregistration
 * - Functional QC steps
 */

nextflow.enable.dsl=2

// Include functional processing modules
include { FUNC_AVERAGE_TMEAN } from '../modules/functional.nf'
include { FUNC_SLICE_TIMING } from '../modules/functional.nf'
include { FUNC_MOTION_CORRECTION } from '../modules/functional.nf'
include { FUNC_GENERATE_TMEAN } from '../modules/functional.nf'
include { FUNC_DESPIKE } from '../modules/functional.nf'
include { FUNC_WITHIN_SES_COREG } from '../modules/functional.nf'
include { FUNC_BIAS_CORRECTION } from '../modules/functional.nf'
include { FUNC_COMPUTE_CONFORM } from '../modules/functional.nf'
include { FUNC_COMPUTE_BRAIN_MASK } from '../modules/functional.nf'
include { FUNC_COMPUTE_REGISTRATION } from '../modules/functional.nf'
include { FUNC_APPLY_CONFORM } from '../modules/functional.nf'
include { FUNC_APPLY_TRANSFORMS } from '../modules/functional.nf'
include { FUNC_APPLY_TRANSFORMS as FUNC_APPLY_TRANSFORMS_MASK } from '../modules/functional.nf'

// Include functional QC modules
include { QC_MOTION_CORRECTION } from '../modules/qc.nf'
include { QC_CONFORM_FUNC } from '../modules/qc.nf'
include { QC_SKULLSTRIPPING_FUNC } from '../modules/qc.nf'
include { QC_REGISTRATION_FUNC } from '../modules/qc.nf'
include { QC_REGISTRATION_FUNC as QC_REGISTRATION_FUNC_INTERMEDIATE } from '../modules/qc.nf'
include { QC_WITHIN_SES_COREG } from '../modules/qc.nf'
include { FUNC_COMPUTE_TSNR } from '../modules/functional.nf'
include { FUNC_TSNR_SESSION_AVERAGE } from '../modules/functional.nf'
include { FUNC_TSNR_SURF_PROJECTION } from '../modules/functional.nf'
include { QC_TSNR } from '../modules/qc.nf'
include { FUNC_COMPUTE_CONFOUNDS } from '../modules/functional.nf'
include { QC_CONFOUNDS } from '../modules/qc.nf'

// Load external Groovy files
def channelHelpers = evaluate(new File("${projectDir}/workflows/channel_helpers.groovy").text)
def funcChannels = evaluate(new File("${projectDir}/workflows/functional_channels.groovy").text)

// Load parameter resolver
def paramResolver = evaluate(new File("${projectDir}/workflows/param_resolver.groovy").text)

// Load config helpers
def configHelpers = evaluate(new File("${projectDir}/workflows/config_helpers.groovy").text)

workflow FUNC_WF {
    take:
    anat_after_bias_brain  // channel from anatomical workflow (Phase 1 final output - brain version for registration)
    anat_reg_transforms  // channel from anatomical workflow
    anat_reg_reference  // channel from anatomical workflow (target_final.nii.gz from ANAT_REGISTRATION)
    surf_actual_subject_id  // [sub, ses, fastsurfer_dir_name] from surface recon; empty when surf recon skipped
    anat_skull_seg      // [sub, ses, T1w-space segmentation] (for confounds tissue regressors)
    anat_skull_seg_lut  // [sub, ses, atlas LUT tsv]
    gpu_queue

    main:
    // ============================================
    // INITIALIZATION
    // ============================================
    // Initialize parameter resolver (if not already initialized in main.nf)
    // This is safe to call multiple times - it will only load configs once
    configHelpers.ensureParamResolverInitialized(paramResolver, params, projectDir)
    
    // Use effective config file (generated in main.nf)
    // This contains all resolved parameters: CLI params → YAML config → defaults.yaml
    // Get path as string - Nextflow processes can accept string paths for 'path' inputs
    def config_file_path = configHelpers.getEffectiveConfigPath(params, projectDir)
    // Pass string path directly - Nextflow will convert to file object when process executes
    // This avoids early validation issues with file() in workflow blocks
    def config_file = config_file_path
    
    // ============================================
    // RESOLVE PARAMETERS (for workflow logic)
    // ============================================
    // Priority: CLI params → YAML config → defaults.yaml
    // All defaults come from defaults.yaml - no hardcoded values
    
    // Resolve YAML-only boolean parameters
    // All defaults come from defaults.yaml
    def func_slice_timing_enabled = paramResolver.getYamlBool("func.slice_timing_correction.enabled")
    def func_motion_correction_enabled = paramResolver.getYamlBool("func.motion_correction.enabled")
    def func_despike_enabled = paramResolver.getYamlBool("func.despike.enabled")
    def func_bias_correction_enabled = paramResolver.getYamlBool("func.bias_correction.enabled")
    def func_conform_enabled = paramResolver.getYamlBool("func.conform.enabled")
    def func_skullstripping_enabled = paramResolver.getYamlBool("func.skullstripping.enabled")
    def func_confounds_enabled = paramResolver.getYamlBool("func.confounds.enabled")
    def func_coreg_runs_within_session = paramResolver.getYamlBool("func.coreg_runs_within_session")
    def registration_enabled = paramResolver.getYamlBool("registration.enabled", true)
    
    // Helper functions
    def isT1wFile = channelHelpers.isT1wFile
    def passThroughFunc = channelHelpers.passThroughFunc
    def findUnmatched = channelHelpers.findUnmatchedFunc
    
    // ============================================
    // PARSE FUNCTIONAL JOBS
    // ============================================
    // Parse functional jobs JSON into channel
    // Channel structure: [sub, ses, run_identifier, file_obj, bids_name]
    def func_jobs_file = file("${params.output_dir}/nextflow_reports/functional_jobs.json")
    if (!new File(func_jobs_file.toString()).exists()) {
        error "Discovery file not found: ${func_jobs_file}\n" +
              "Please run the discovery script before starting Nextflow."
    }
    
    def func_jobs_ch = Channel.fromPath(func_jobs_file)
        .splitJson()
        .map { job ->
            def sub = job.subject_id.toString()
            def ses = job.session_id ? job.session_id.toString() : null
            def file_obj = file(job.file_path as String)
            def bids_name = channelHelpers.normalizeBidsNiftiTemplate(file_obj.toString())
            def run_identifier = channelHelpers.extractRunIdentifier(bids_name)
            [sub, ses, run_identifier, file_obj, bids_name]
        }
    
    // ============================================
    // FUNCTIONAL PREPROCESSING PIPELINE
    // ============================================
    // Sequential processing: slice timing → motion correction → despike
    // Channel structure maintained: [sub, ses, run_identifier, bold_file, tmean_file, bids_name]
    
    // ============================================
    // SLICE_TIMING
    // ============================================
    def func_after_slice = func_jobs_ch
    if (func_slice_timing_enabled) {
        FUNC_SLICE_TIMING(func_jobs_ch, config_file)
        func_after_slice = FUNC_SLICE_TIMING.out.output
    } else {
        func_after_slice = func_jobs_ch.map(passThroughFunc)
    }

    def func_after_slice_passthrough = func_after_slice.map(passThroughFunc)
    
    // ============================================
    // MOTION_CORRECTION
    // ============================================
    def func_after_motion = func_after_slice_passthrough
    def func_motion_params = Channel.empty()
    if (func_motion_correction_enabled) {
        FUNC_MOTION_CORRECTION(func_after_slice_passthrough, config_file)
        func_after_motion = FUNC_MOTION_CORRECTION.out.output
        func_motion_params = FUNC_MOTION_CORRECTION.out.motion_params
    } else {
        FUNC_GENERATE_TMEAN(func_after_slice_passthrough, config_file)
        func_after_motion = FUNC_GENERATE_TMEAN.out.output
    }
    
    // ============================================
    // DESPIKE
    // ============================================
    def func_after_despike = func_after_motion
    if (func_despike_enabled) {
        FUNC_DESPIKE(func_after_motion, config_file)
        func_after_despike = FUNC_DESPIKE.out.output
    } else {
        func_after_despike = func_after_motion
    }
    
    // ============================================
    // WITHIN-SESSION COREGISTRATION
    // ============================================
    // Coregister multiple runs within the same session to a reference run
    // Input: func_after_despike: [sub, ses, run_id, bold_file, tmean_file, bids_name]
    // Output: func_after_coreg: [sub, ses, run_id, bold_file, tmean_file, bids_name]
    def func_after_coreg = func_after_despike
    def func_coreg_transforms_ch = Channel.empty()
    def func_tmean_averaged_ch = Channel.empty()
    def func_coreg_success = false
    def func_single_run_ses = Channel.empty()  // Initialize for use in compute phase
    
    if (func_coreg_runs_within_session) {
        def coregChannels = funcChannels.prepareWithinSessionCoregChannels(func_after_despike, Channel)
        
        coregChannels.func_later_runs
            .multiMap { sub, ses, run_identifier, bold, tmean, bids_name, ref_tmean, ref_run_identifier ->
                combined: [sub, ses, run_identifier, bold, tmean, bids_name]
                reference: ref_tmean
                ref_run_identifier_val: ref_run_identifier
            }
            .set { func_coreg_multi }
        
        // Use GPU token only when workflow-level GPU scheduling is enabled.
        def use_coreg_gpu = params.use_gpu
        def coreg_gpu_input = use_coreg_gpu ? gpu_queue : Channel.value('none')

        FUNC_WITHIN_SES_COREG(func_coreg_multi.combined, func_coreg_multi.reference, func_coreg_multi.ref_run_identifier_val, config_file, coreg_gpu_input)
        if (use_coreg_gpu) {
            FUNC_WITHIN_SES_COREG.out.gpu_token.subscribe { gpu_queue << it }
        }
        func_coreg_transforms_ch = FUNC_WITHIN_SES_COREG.out.transforms
        
        // Separate multi-run sessions (need averaging) from single-run sessions (skip averaging)
        // Multi-run sessions: first runs + coregistered later runs
        def func_multi_run_ses = coregChannels.func_first_runs
            .mix(FUNC_WITHIN_SES_COREG.out.output)
        
        // Single-run sessions: pass through unchanged (no averaging needed)
        // Store in outer scope for use in compute phase
        func_single_run_ses = coregChannels.func_single_run_ses
        
        // Combine all runs for func_after_coreg (for anatomical selection and apply phase)
        def func_all_coreg = func_multi_run_ses
            .mix(func_single_run_ses)
        
        // Only average tmean for multi-run sessions
        def func_for_averaging_ch = func_multi_run_ses
            .groupTuple(by: [0, 1])
            .map { sub, ses, run_identifier_list, bold_list, tmean_list, bids_list ->
                def tmean_paths = tmean_list.collect { file -> file.toString() }
                def tmean_paths_json = groovy.json.JsonOutput.toJson(tmean_paths)
                def bids_name = bids_list[0]
                [sub, ses, tmean_paths_json, bids_name]
            }
        
        func_for_averaging_ch
            .multiMap { sub, ses, tmean_paths, bids_name ->
                tmean_files_input: tmean_paths
                subject_id_input: sub
                session_id_input: ses
                bids_name_input: bids_name
            }
            .set { averagingChannels }
        
        FUNC_AVERAGE_TMEAN(
            averagingChannels.tmean_files_input,
            averagingChannels.subject_id_input,
            averagingChannels.session_id_input,
            averagingChannels.bids_name_input,
            config_file
        )
        
        // For single-run sessions, create channel matching FUNC_AVERAGE_TMEAN output format
        // but with original tmean and bids_name (no desc-coreg)
        // Format: [sub, ses, tmean, bids_name]
        def func_single_run_tmean_ch = func_single_run_ses
            .map { sub, ses, run_identifier, bold, tmean, bids_name ->
                [sub, ses, tmean, bids_name]
            }
        
        // Combine averaged tmean (multi-run) with original tmean (single-run)
        func_tmean_averaged_ch = FUNC_AVERAGE_TMEAN.out.output
            .mix(func_single_run_tmean_ch)
        
        func_after_coreg = func_all_coreg
        func_coreg_success = true
        
        // QC only for multi-run sessions (single-run sessions don't need coreg QC)
        def func_coreg_qc_input = funcChannels.prepareCoregQCChannels(coregChannels.func_first_runs, FUNC_AVERAGE_TMEAN.out.output)
        QC_WITHIN_SES_COREG(func_coreg_qc_input, config_file)
        // Note: QC_WITHIN_SES_COREG metadata is collected in the COLLECT QC CHANNELS section below
    }
    
    // ============================================
    // ANATOMICAL SELECTION
    // ============================================
    // Select appropriate anatomical reference for each functional session
    // Priority: 1) Same session, 2) Different session (same subject), 3) Dummy (no anatomical)
    // Note: All runs in the same session use the same anatomical reference
    // Input: func_after_coreg: [sub, ses, run_id, bold_file, tmean_file, bids_name]
    //        anat_after_bias_brain: [sub, ses, brain_file, bids_name] (Phase 1 final output - brain version)
    // Output: [sub, ses, anat_file, anat_ses] (session-level only, no run_id)
    def dummy_anat = file("${workDir}/dummy_anat.dummy")
    dummy_anat.toFile().text = ""
    def func_anat_selection = channelHelpers.performFuncAnatomicalSelection(
        func_after_coreg,
        anat_after_bias_brain,
        isT1wFile,
        findUnmatched,
        dummy_anat
    )
    // func_anat_selection: [sub, ses, anat_file, anat_ses] (session-level)

    // // debug print
    // func_anat_selection.view() {
    //     println "|| func_anat_selection ||: ${it}"
    // }

    // ============================================
    // COMPUTE PHASE
    // ============================================
    // Compute transforms and masks on tmean (session-level or per-run)
    // Two paths: session-level (when coreg enabled for multi-run) vs per-run processing
    def func_compute_conform_output = Channel.empty()
    def func_compute_conform_transforms = Channel.empty()
    def func_compute_mask_output = Channel.empty()
    def func_compute_reg_output = Channel.empty()
    
    // Determine processing mode and setup helper variables
    // When coreg is enabled:
    //   - Multi-run sessions: use session-level processing (empty run_id) with averaged tmean
    //   - Single-run sessions: use per-run processing (preserve run_id) with original tmean
    // When coreg is disabled: use per-run processing for all sessions
    // Note: isSessionLevel is used for channel operations (combine vs join) - true when coreg enabled
    def isSessionLevel = func_coreg_runs_within_session && func_coreg_success
    def computeInput
    if (isSessionLevel) {
        // Multi-run sessions: session-level processing with averaged tmean
        def computeInput_multi_run = FUNC_AVERAGE_TMEAN.out.output
            .map { sub, ses, tmean, bids_name -> [sub, ses, "", tmean, bids_name] }
        
        // Single-run sessions: per-run processing with original tmean (preserve run_id)
        def computeInput_single_run = func_single_run_ses
            .map { sub, ses, run_id, bold, tmean, bids_name -> [sub, ses, run_id, tmean, bids_name] }
        
        // Combine multi-run and single-run sessions
        computeInput = computeInput_multi_run.mix(computeInput_single_run)
    } else {
        // Coreg disabled: per-run processing for all sessions
        computeInput = func_after_coreg.map { sub, ses, run_id, bold, tmean, bids_name -> [sub, ses, run_id, tmean, bids_name] }
    }
    
    // ============================================
    // BIAS_CORRECTION
    // ============================================
    def func_after_bias = computeInput
    if (func_bias_correction_enabled) {
        FUNC_BIAS_CORRECTION(computeInput, config_file)
        func_after_bias = FUNC_BIAS_CORRECTION.out.output
    } else {
        func_after_bias = computeInput
    }
    
    // ============================================
    // COMPUTE CONFORM
    // ============================================
    if (func_conform_enabled) {
        // Combine with anatomical selection (session-level)
        // func_after_bias: [sub, ses, run_id, tmean, bids_name] (run_id may be "" for session-level)
        // func_anat_selection: [sub, ses, anat_file, anat_ses] (session-level)
        // Use combine() since anatomical selection is session-level and may match multiple runs
        def func_conform_with_anat = func_after_bias
            .combine(func_anat_selection, by: [0, 1])  // Combine by [sub, ses]
            .map { sub, ses, run_id, tmean, bids_name, anat_file, anat_ses ->
                [sub, ses, run_id, tmean, bids_name, anat_file]
            }
        
        func_conform_with_anat
            .multiMap { sub, ses, run_id, tmean, bids_name, anat_file ->
                combined: [sub, ses, run_id, tmean, bids_name]
                reference: anat_file
            }
            .set { func_compute_conform_multi }

        FUNC_COMPUTE_CONFORM(func_compute_conform_multi.combined, func_compute_conform_multi.reference, config_file)
        func_compute_conform_output = FUNC_COMPUTE_CONFORM.out.output
        func_compute_conform_transforms = FUNC_COMPUTE_CONFORM.out.transforms
    } else {
        func_compute_conform_output = func_after_bias
            .map { sub, ses, run_id, tmean, bids_name ->
                [sub, ses, run_id, tmean, bids_name]
            }
        def dummy_forward_transform = file("${workDir}/dummy_conform_forward_transform.dummy")
        dummy_forward_transform.toFile().text = ""
        def dummy_inverse_transform = file("${workDir}/dummy_conform_inverse_transform.dummy")
        dummy_inverse_transform.toFile().text = ""
        func_compute_conform_transforms = func_after_bias
            .map { sub, ses, run_id, tmean, bids_name ->
                [sub, ses, run_id, dummy_forward_transform, dummy_inverse_transform]
            }
    }
    
    // ============================================
    // COMPUTE BRAIN MASK
    // ============================================
    if (func_skullstripping_enabled) {
        def func_compute_mask_input = func_compute_conform_output
            .map { sub, ses, run_id, conformed_tmean, bids_name ->
                [sub, ses, run_id, conformed_tmean, bids_name]
            }
        
        // Use GPU token only when workflow-level GPU scheduling is enabled (use_gpu).
        // Without this gate, brain-mask always pulls gpu_id=0 from gpu_queue and runs on
        // GPU even in CPU mode (general.gpu_device=-1 / use_gpu=false).
        def use_mask_gpu = params.use_gpu
        def mask_gpu_input = use_mask_gpu ? gpu_queue : Channel.value('none')
        FUNC_COMPUTE_BRAIN_MASK(func_compute_mask_input, config_file, mask_gpu_input)
        if (use_mask_gpu) {
            FUNC_COMPUTE_BRAIN_MASK.out.gpu_token.subscribe { gpu_queue << it }
        }
        func_compute_mask_output = FUNC_COMPUTE_BRAIN_MASK.out.output
    } else {
        func_compute_mask_output = func_compute_conform_output
            .map { sub, ses, run_id, conformed_tmean, bids_name ->
                def dummy_mask = file("${workDir}/dummy_brain_mask.dummy").tap { it.toFile().text = "" }
                [sub, ses, run_id, conformed_tmean, bids_name, dummy_mask]
            }
    }
    
    // ============================================
    // COMPUTE REGISTRATION
    // ============================================
    // Compute registration to target space
    // Input: func_compute_mask_output: [sub, ses, run_id, masked_tmean, bids_name, mask]
    // Output: func_compute_reg_output: [sub, ses, run_id, registered_tmean, bids_name, anat_ses]
    if (registration_enabled) {
        def func_compute_reg_input = func_compute_mask_output
            .map { sub, ses, run_id, masked_tmean, bids_name, mask ->
                [sub, ses, run_id, masked_tmean, bids_name]
            }
            .combine(func_anat_selection, by: [0, 1])  // Combine by [sub, ses] - anatomical selection is session-level
            .map { sub, ses, run_id, masked_tmean, bids_name, anat_file, anat_ses ->
                [sub, ses, run_id, masked_tmean, bids_name, anat_file, anat_ses]
            }

        func_compute_reg_input
            .multiMap { sub, ses, run_id, masked_tmean, bids_name, anat_file, anat_ses ->
                combined: [sub, ses, run_id, masked_tmean, bids_name, anat_ses]
                reference: anat_file
            }
            .set { func_compute_reg_multi }

        def use_registration_gpu = params.use_gpu
        def gpu_input = use_registration_gpu ? gpu_queue : Channel.value('none')

        FUNC_COMPUTE_REGISTRATION(func_compute_reg_multi.combined, func_compute_reg_multi.reference, config_file, gpu_input)
        if (use_registration_gpu) {
            FUNC_COMPUTE_REGISTRATION.out.gpu_token.subscribe { gpu_queue << it }
        }
        func_compute_reg_output = FUNC_COMPUTE_REGISTRATION.out.output
        func_reg_transforms = FUNC_COMPUTE_REGISTRATION.out.transforms
        func_reg_reference = FUNC_COMPUTE_REGISTRATION.out.reference
    } else {
        func_compute_reg_output = func_compute_mask_output
            .map { sub, ses, run_id, masked_tmean, bids_name, mask ->
                def dummy_transform = file("${workDir}/dummy_reg_transform.dummy").tap { it.toFile().text = "" }
                [sub, ses, run_id, masked_tmean, bids_name, dummy_transform, ""]
            }
    }
    
    // ============================================
    // APPLY PHASE
    // ============================================
    // Apply computed transforms to full BOLD 4D data
    // Input: func_after_coreg: [sub, ses, run_id, bold_file, tmean_file, bids_name]
    // Output: func_apply_reg: [sub, ses, run_id, registered_bold, registered_boldref, bids_name]
    def func_apply_bold_input = func_after_coreg
    
    // ============================================
    // APPLY CONFORM
    // ============================================
    // Apply conform transform to BOLD data
    // Input: func_apply_bold_input: [sub, ses, run_id, bold_file, tmean_file, bids_name]
    //        func_compute_conform_output: [sub, ses, run_id, conformed_tmean, bids_name]
    //        func_compute_conform_transforms: [sub, ses, run_id, forward_xfm, inverse_xfm]
    // Output: [sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name]
    def func_apply_conform_output = Channel.empty()
    if (func_conform_enabled) {
        // Step 1: Extract forward transform
        // Input: func_compute_conform_transforms: [sub, ses, run_id, forward_xfm, inverse_xfm]
        // Output: [sub, ses, run_id, forward_xfm] (or [sub, ses, forward_xfm] for session-level)
        def func_conform_forward_xfm = isSessionLevel ?
            func_compute_conform_transforms.map { sub, ses, run_id, func2target_xfm, inverse_transform -> [sub, ses, func2target_xfm] } :
            func_compute_conform_transforms.map { sub, ses, run_id, func2target_xfm, inverse_transform -> [sub, ses, run_id, func2target_xfm] }
        
        // Step 2: Join/combine BOLD input with conformed tmean
        // Session-level: combine by [sub, ses] (one session-level item combines with all runs)
        // Per-run: join by [sub, ses, run_id] (exact match per run)
        def func_apply_with_conformed = isSessionLevel ?
            func_apply_bold_input.combine(func_compute_conform_output, by: [0, 1]) :
            func_apply_bold_input.join(func_compute_conform_output, by: [0, 1, 2])
        
        // Step 3: Join/combine with forward transform
        def func_apply_conform_input = isSessionLevel ?
            func_apply_with_conformed.combine(func_conform_forward_xfm, by: [0, 1])
                .map { sub, ses, run_id, bold, tmean, bids_name, run_id2, conformed_tmean, bids_name2, func2target_xfm ->
                    [sub, ses, run_id, bold, func2target_xfm, conformed_tmean, bids_name]
                } :
            func_apply_with_conformed.join(func_conform_forward_xfm, by: [0, 1, 2])
                .map { sub, ses, run_id, bold, tmean, bids_name, conformed_tmean, bids_name2, func2target_xfm ->
                    [sub, ses, run_id, bold, func2target_xfm, conformed_tmean, bids_name]
                }
        
        FUNC_APPLY_CONFORM(func_apply_conform_input, config_file)
        func_apply_conform_output = FUNC_APPLY_CONFORM.out.output
    } else {
        // Pass through real bold and tmean so downstream QC (e.g. QC_SKULLSTRIPPING_FUNC) has a valid underlay
        func_apply_conform_output = func_apply_bold_input
            .map { sub, ses, run_id, bold, tmean, bids_name ->
                [sub, ses, run_id, bold, tmean, bids_name]
            }
    }
    
    // ============================================
    // APPLY REGISTRATION
    // ============================================
    // Registration application phase: Apply computed transforms to BOLD data
    // Input: func_apply_conform_output: [sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name]
    // Output: func_apply_reg: [sub, ses, run_id, registered_bold, registered_boldref, bids_name]
    def func_apply_reg = func_apply_conform_output
    def func_apply_reg_reference = Channel.empty()
    // Confounds TSV per run, for the motion QC figure's flagged-frame shading. Declared out here
    // because `def` inside the blocks below is not visible at the QUALITY CONTROL section, and it
    // stays empty when registration or confounds are disabled.
    def confounds_for_qc = Channel.empty()
    if (registration_enabled) {
        // PREPARE ANATOMICAL REGISTRATION DATA
        // Extract forward transform and join with reference
        // Input: anat_reg_transforms: [sub, ses, anat2template_xfm, inverse_transform]
        //        anat_reg_reference: [sub, ses, ref_from_anat_reg]
        // Output: [sub, ses, anat2template_xfm, ref_from_anat_reg]
        def anat_reg_all_real = anat_reg_transforms
            .map { sub, ses, bids_name, anat2template_xfm, inverse_transform -> [sub, ses, anat2template_xfm] }
            .join(anat_reg_reference, by: [0, 1]) // join by [sub, ses]
        
        // JOIN ANATOMICAL REGISTRATION BY SESSION
        // Match anatomical registration by anatomical session, then map back to functional session
        // Works for both same-session (anat_ses == ses_func) and cross-session (anat_ses != ses_func)
        // 
        // IMPORTANT: We use combine() + filter() instead of join() because multiple functional sessions
        // may reference the SAME anatomical session (e.g., 032309_001 and 032309_002 both use anat from 001).
        // join() only matches one item per key due to channel consumption, causing a race condition.
        // combine() creates all subject-level combinations, then filter keeps only matching anat_ses.
        //
        // Input: func_compute_reg_output: [sub, ses, run_id, registered_tmean, bids_name, anat_ses]
        //        anat_reg_all_real: [sub, ses, anat2template_xfm, ref_from_anat_reg] (keyed by anatomical session)
        // Output: [sub, ses_func, xfm, ref]
        def anat_reg_by_func_ses = func_compute_reg_output
            .map { sub, ses, run_id, registered_tmean, bids_name, anat_ses ->
                [sub, ses, anat_ses]  // Keep func_ses and anat_ses
            }
            .unique { sub, ses, anat_ses -> [sub, ses] }  // Deduplicate by functional session
            .combine(anat_reg_all_real, by: 0)  // Combine by subject only - creates all subject-level combinations
            .filter { sub, ses_func, anat_ses, anat_reg_ses, xfm, ref ->
                // Normalize session identifiers for comparison
                // Handle subject-level case: "" or "null" or null should all match
                // Note: Nextflow may pass "null" as a string when session_id is empty/null in Groovy
                def normalize_ses = { ses_val ->
                    if (ses_val == null || ses_val == "") return null
                    if (ses_val instanceof String && ses_val.toLowerCase() == 'null') return null
                    return ses_val
                }
                def anat_ses_norm = normalize_ses(anat_ses)
                def anat_reg_ses_norm = normalize_ses(anat_reg_ses)
                anat_ses_norm == anat_reg_ses_norm  // Keep only where functional's anat_ses matches anatomical's session
            }
            .map { sub, ses_func, anat_ses, anat_reg_ses, xfm, ref -> 
                [sub, ses_func, xfm, ref]  // Map to final format
            }

        // CREATE DUMMY REGISTRATION FOR MISSING DATA
        // Create dummy registration for functional sessions without anatomical data
        // Input: func_compute_reg_output: [sub, ses, run_id, registered_tmean, bids_name, anat_ses]
        // Output: [sub, ses, dummy_xfm, dummy_ref]
        def dummy_anat2template_xfm = file("${workDir}/dummy_anat2template_xfm.dummy")
        dummy_anat2template_xfm.toFile().text = ""
        def dummy_anat_reg_ref = file("${workDir}/dummy_anat_reg_ref.dummy")
        dummy_anat_reg_ref.toFile().text = ""
        
        def anat_reg_all_dummy = func_compute_reg_output
            .map { sub, ses, run_id, registered_tmean, bids_name, anat_ses -> [sub, ses] }
            .unique() // deduplicate by [sub, ses]
            .map { sub, ses -> [sub, ses, dummy_anat2template_xfm, dummy_anat_reg_ref] }

        // COMBINE ALL ANATOMICAL REGISTRATION DATA
        // Mix all sources, group by session, and select best (prefer real over dummy)
        // Input: anat_reg_by_func_ses (real registration), anat_reg_all_dummy
        // Output: [sub, ses, xfm, ref] where xfm/ref are real files if available, dummy otherwise
        def anat_reg_all = anat_reg_by_func_ses
            .mix(anat_reg_all_dummy) // mix real and dummy registration data
            .map { sub, ses, xfm, ref -> [sub, ses, [xfm, ref]] }  // Wrap for groupTuple
            .groupTuple(by: [0, 1]) // group by [sub, ses]
            .map { sub, ses, entries ->
                // Prefer first non-dummy entry, or first entry if all are dummy
                def selected = entries.find { xfm, ref ->
                    !(xfm.toString().contains('.dummy') || ref.toString().contains('.dummy'))
                } ?: entries[0]
                [sub, ses] + selected // add selected xfm and ref to [sub, ses]
            }

        // PREPARE FUNCTIONAL REGISTRATION DATA FOR APPLICATION
        // Two paths: session-level (run_id == "") vs per-run processing
        // Step 1: Extract forward transform from functional registration
        // Input: func_reg_transforms: [sub, ses, run_id, func2target_xfm, inverse_transform]
        // Output: [sub, ses, run_id, func2target_xfm] (or [sub, ses, "", func2target_xfm] for session-level)
        def func_reg_transforms_forward = func_reg_transforms
            .map { sub, ses, run_id, func2target_xfm, inverse_transform ->
                [sub, ses, run_id, func2target_xfm]
            }
        
        // Step 2: Join/combine functional registration outputs (registered tmean + transform + reference)
        // Input: func_compute_reg_output: [sub, ses, run_id, registered_tmean, bids_name, anat_ses]
        //        func_reg_transforms_forward: [sub, ses, run_id, func2target_xfm]
        //        func_reg_reference: [sub, ses, run_id, ref_from_func_reg]
        // Output: [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg]
        // Note: Drop anat_ses - not needed after join with anat_reg_all
        def func_reg_with_transform = isSessionLevel ?
            func_compute_reg_output.combine(func_reg_transforms_forward, by: [0, 1]) :
            func_compute_reg_output.join(func_reg_transforms_forward, by: [0, 1, 2])
        
        // Join with reference - ensure all entries have ref_from_func_reg
        // FUNC_COMPUTE_REGISTRATION always emits a real file (anatomical brain or template), never a dummy
        // If a dummy file appears here, it indicates a channel join/mapping bug
        def func_reg_with_ref = isSessionLevel ?
            func_reg_with_transform.combine(func_reg_reference, by: [0, 1])
                .map { sub, ses, run_id, registered_tmean, bids_name, anat_ses, run_id2, func2target_xfm, run_id3, ref_from_func_reg ->
                    // Validate ref_from_func_reg is not a dummy file
                    // This should never happen - FUNC_COMPUTE_REGISTRATION always emits real files
                    if (ref_from_func_reg.toString().contains('.dummy')) {
                        error "ERROR: ref_from_func_reg is a dummy file for ${sub}/${ses}. " +
                              "FUNC_COMPUTE_REGISTRATION should have emitted a real reference file (anatomical brain or template). " +
                              "This indicates a channel join/mapping bug. Check that FUNC_COMPUTE_REGISTRATION.out.reference " +
                              "has entries for all runs and that the join keys match correctly."
                    }
                    [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg]
                } :
            func_reg_with_transform.join(func_reg_reference, by: [0, 1, 2])
                .map { sub, ses, run_id, registered_tmean, bids_name, anat_ses, func2target_xfm, ref_from_func_reg ->
                    // Validate ref_from_func_reg is not a dummy file
                    // This should never happen - FUNC_COMPUTE_REGISTRATION always emits real files
                    if (ref_from_func_reg.toString().contains('.dummy')) {
                        error "ERROR: ref_from_func_reg is a dummy file for ${sub}/${ses}/${run_id}. " +
                              "FUNC_COMPUTE_REGISTRATION should have emitted a real reference file (anatomical brain or template). " +
                              "This indicates a channel join/mapping bug. Check that FUNC_COMPUTE_REGISTRATION.out.reference " +
                              "has entries for all runs and that the join keys [sub, ses, run_id] match correctly."
                    }
                    [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg]
                }

        // Step 3: Combine with anatomical registration data
        // Input: func_reg_with_ref: [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg]
        //        anat_reg_all: [sub, ses, anat2template_xfm, ref_from_anat_reg]
        // Output: [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
        def func_reg_with_anat = func_reg_with_ref
            .combine(anat_reg_all, by: [0, 1])
            .map { sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg ->
                [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
            }

        // Step 4: Join/combine with conformed BOLD data for application
        // Input: func_apply_conform_output: [sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name]
        //        func_reg_with_anat: [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
        // Output: [sub, ses, run_id, conformed_bold, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
        def func_apply_reg_with_bold = isSessionLevel ?
            func_apply_conform_output.combine(func_reg_with_anat, by: [0, 1])
                .map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name, run_id2, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg ->
                    [sub, ses, run_id, conformed_bold, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
                } :
            func_apply_conform_output.join(func_reg_with_anat, by: [0, 1, 2])
                .map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg ->
                    [sub, ses, run_id, conformed_bold, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
                }

        // SPLIT CHANNELS FOR FUNC_APPLY_TRANSFORMS
        // Split into registration parameters and BOLD file for process input
        // Input: func_apply_reg_with_bold: [sub, ses, run_id, conformed_bold, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
        // Output channels:
        //   - reg_combined: [sub, ses, run_id, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
        //   - func_4d_file: conformed_bold (separate channel for 4D file)
        func_apply_reg_with_bold
            .multiMap { sub, ses, run_id, conformed_bold, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg ->
                reg_combined: [sub, ses, run_id, bids_name, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
                func_4d_file: conformed_bold
            }
            .set { func_apply_reg_multi }
        
        // // debug print
        // func_apply_reg_multi.reg_combined.view() {
        //     println "|| func_apply_reg_multi.reg_combined ||: ${it}"
        // }
        
        // Apply transforms to BOLD data
        // Output: func_apply_reg: [sub, ses, run_id, registered_bold, registered_boldref, bids_name]
        //         func_apply_reg_reference: [sub, ses, run_id, target_final] (for QC)
        FUNC_APPLY_TRANSFORMS(func_apply_reg_multi.reg_combined, func_apply_reg_multi.func_4d_file, "bold", config_file)
        func_apply_reg = FUNC_APPLY_TRANSFORMS.out.output
        func_apply_reg_reference = FUNC_APPLY_TRANSFORMS.out.reference
        func_apply_reg_intermediate = FUNC_APPLY_TRANSFORMS.out.intermediate_output
        
        // ============================================
        // APPLY REGISTRATION TO BRAIN MASK
        // ============================================
        // Transform brain mask using the same transforms applied to BOLD data
        // Reuse func_reg_with_anat and apply transforms with data_type="mask"
        // Input: func_compute_mask_output: [sub, ses, run_id, masked_tmean, bids_name, brain_mask] (mask in conformed space)
        //        func_reg_with_anat: [sub, ses, run_id, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
        // Output: func_brain_mask_registered: [sub, ses, run_id, transformed_mask, transformed_mask (dup), bids_name] (in template space)
        if (func_skullstripping_enabled) {
            // Extract mask from func_compute_mask_output
            // Use original BOLD bids_name as bids_naming_template (not mask file path)
            // The mask file path may have wrong entities; we'll construct the correct mask name from BOLD bids_name
            def func_mask_for_transform = func_compute_mask_output
                .map { sub, ses, run_id, masked_tmean, bids_name, mask ->
                    [sub, ses, run_id, mask, bids_name]  // mask file as input, original BOLD bids_name as template
                }
            
            // Join mask with transform data (same pattern as BOLD)
            // When session-level: mask has run_id="", need to expand to all runs by combining with func_apply_conform_output
            // When per-run: mask has actual run_id, join directly with func_reg_with_anat
            def func_mask_with_transforms = isSessionLevel ?
                // Step 1: Expand mask to all runs by combining with func_apply_conform_output (which has all run_ids)
                func_mask_for_transform
                    .combine(func_apply_conform_output.map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name -> [sub, ses, run_id, bids_name] }, by: [0, 1])
                    .map { sub, ses, run_id_mask, mask, mask_bids, run_id_bold, bids_name_bold ->
                        // Use the run_id from BOLD (actual run), and bids_name from BOLD (per-run bids_name)
                        [sub, ses, run_id_bold, mask, bids_name_bold]
                    }
                    // Step 2: Combine with func_reg_with_anat (session-level, run_id="") to get transforms
                    .combine(func_reg_with_anat, by: [0, 1])
                    .map { sub, ses, run_id, mask, mask_bids, run_id2, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg ->
                        [sub, ses, run_id, mask_bids, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg, mask]
                    } :
                func_mask_for_transform
                    .join(func_reg_with_anat, by: [0, 1, 2])
                    .map { sub, ses, run_id, mask, mask_bids, registered_tmean, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg ->
                        [sub, ses, run_id, mask_bids, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg, mask]
                    }
            
            // Split into registration parameters and mask file (same pattern as BOLD)
            func_mask_with_transforms
                .multiMap { sub, ses, run_id, mask_bids, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg, mask ->
                    reg_combined: [sub, ses, run_id, mask_bids, func2target_xfm, ref_from_func_reg, anat2template_xfm, ref_from_anat_reg]
                    mask_file: mask
                }
                .set { func_mask_multi }
            
            // Apply transforms to mask (single call - process handles sequential internally)
            FUNC_APPLY_TRANSFORMS_MASK(func_mask_multi.reg_combined, func_mask_multi.mask_file, "mask", config_file)
            func_brain_mask_registered = FUNC_APPLY_TRANSFORMS_MASK.out.output
        }

        // ============================================
        // TSNR COMPUTATION
        // ============================================
        // bold may be List or single Path (sequential reg stages multiple *space-*desc-*.nii.gz in work dir).
        // Pick one BOLD + space_label here; mask matched by space token in join below (no Python staging resolve).
        def tsnr_base = func_apply_reg
            .map { sub, ses, run_id, bold, boldref, bids_name ->
                def files = (bold instanceof List) ? bold : [bold]
                def t1w = files.find { f ->
                    "${f}".contains('space-T1w') && "${f}".contains('desc-preproc') && "${f}".endsWith('_bold.nii.gz')
                }
                def tmpl = files.find { f ->
                    "${f}".contains('space-') && !"${f}".contains('space-T1w') && "${f}".contains('desc-preproc') && "${f}".endsWith('_bold.nii.gz')
                }
                def other = files.find { f ->
                    !"${f}".contains('space-') && "${f}".contains('desc-preproc') && "${f}".endsWith('_bold.nii.gz')
                }
                def bold_sel = t1w ?: tmpl ?: other ?: files.last()
                def m = ("${bold_sel}" =~ /space-([^_]+)/)
                def space_token = m.find() ? m.group(1) : null
                def space_label = (space_token == 'T1w') ? 'T1w' : (space_token ? 'template' : 'other')
                [sub, ses, run_id, bold_sel, bids_name, space_label, space_token]
            }

        def tsnr_run_input
        if (func_skullstripping_enabled) {
            tsnr_run_input = tsnr_base
                .join(
                    func_brain_mask_registered
                        .map { sub, ses, run_id, mask, _dup, bids_name -> [sub, ses, run_id, mask] },
                    by: [0, 1, 2]
                )
                .map { sub, ses, run_id, bold_sel, bids_name, space_label, space_token, mask ->
                    def mfiles = (mask instanceof List) ? mask : [mask]
                    def mask_sel = space_token
                        ? (mfiles.find { f -> "${f}".contains("space-${space_token}") } ?: mfiles.last())
                        : mfiles.last()
                    [sub, ses, run_id, bold_sel, bids_name, mask_sel, space_label]
                }
        } else {
            tsnr_run_input = tsnr_base
                .map { sub, ses, run_id, bold_sel, bids_name, space_label, space_token ->
                    def dm = file("${workDir}/dummy_brain_mask_tsnr.dummy").tap { it.toFile().text = "" }
                    [sub, ses, run_id, bold_sel, bids_name, dm, space_label]
                }
        }

        FUNC_COMPUTE_TSNR(tsnr_run_input, config_file)

        def tsnr_grouped = FUNC_COMPUTE_TSNR.out.output
            .filter { sub, ses, run_id, tsnr_nii, bids_name, bold_space -> tsnr_nii && !"${tsnr_nii}".endsWith('.dummy') }
            .groupTuple(by: [0, 1])
            .map { sub, ses, run_ids, tsnr_files, bids_names, spaces ->
                def session_space = spaces.every { it == 'T1w' } ? 'T1w' : spaces[0]
                [sub, ses, run_ids[0], groovy.json.JsonOutput.toJson(tsnr_files*.toString()), bids_names[0], session_space]
            }

        FUNC_TSNR_SESSION_AVERAGE(tsnr_grouped, config_file)

        def surf_recon_enabled = paramResolver.getYamlBool("anat.surface_reconstruction.enabled")
        def tsnr_t1w_ch = FUNC_TSNR_SESSION_AVERAGE.out.output.filter { sub, ses, avg, bids, space -> space == 'T1w' }
        def tsnr_other_ch = FUNC_TSNR_SESSION_AVERAGE.out.output.filter { sub, ses, avg, bids, space -> space != 'T1w' }

        def tsnr_qc_input
        // Separate LH/RH dummy paths so Nextflow does not collide staging two path() inputs with the same basename.
        def dummy_sf_lh = file("${workDir}/dummy_surf_tsnr_lh.dummy").tap { it.toFile().text = "" }
        def dummy_sf_rh = file("${workDir}/dummy_surf_tsnr_rh.dummy").tap { it.toFile().text = "" }

        if (surf_recon_enabled) {
            def surf_actual_subject_id_logged = surf_actual_subject_id

            // Subject-level aid (no _ses- in name) -> broadcast to all T1w sessions of that subject.
            def surf_id_subject_level = surf_actual_subject_id_logged
                .filter { sub, ses, aid -> !("${aid}".contains('_ses-')) }
                .map { sub, ses, aid -> [sub, aid] }

            // Session-level aid (has _ses-) -> re-key by the session encoded in aid itself.
            def surf_id_session_level = surf_actual_subject_id_logged
                .filter { sub, ses, aid -> "${aid}".contains('_ses-') }
                .map { sub, ses, aid ->
                    def m = ("${aid}" =~ /_ses-([^_]+)/)
                    def ses_from_aid = m.find() ? m.group(1) : ses
                    [sub, ses_from_aid, aid]
                }

            def func_session_keys = tsnr_t1w_ch
                .map { sub, ses, avg, bids, space -> [sub, ses] }
                .unique()

            def surf_id_broadcast = func_session_keys
                .combine(surf_id_subject_level, by: 0)
                .map { sub, ses, aid -> [sub, ses, aid] }

            def surf_id_normalized = surf_id_session_level.mix(surf_id_broadcast)

            // remainder: true emits surf-only keys as short tuples (no tSNR row); skip those before destructuring.
            def tsnr_t1w_with_subid = tsnr_t1w_ch
                .join(surf_id_normalized, by: [0, 1], remainder: true)
                .filter { row -> row.size() == 6 }
                .map { sub, ses, avg, bids, space, actual_subid -> [sub, ses, avg, bids, space, actual_subid ?: ''] }

            def tsnr_t1w_has_surf = tsnr_t1w_with_subid.filter { sub, ses, avg, bids, space, aid -> aid }
            def tsnr_t1w_no_surf = tsnr_t1w_with_subid.filter { sub, ses, avg, bids, space, aid -> !aid }

            FUNC_TSNR_SURF_PROJECTION(
                tsnr_t1w_has_surf.map { sub, ses, avg, bids, space, aid -> [sub, ses, avg, bids, aid] },
                config_file
            )
            def t1w_with_surf = tsnr_t1w_has_surf
                .join(FUNC_TSNR_SURF_PROJECTION.out.output, by: [0, 1])
                .map { sub, ses, avg, bids, space, aid, lh_gii, rh_gii, _bids2 ->
                    [sub, ses, avg, lh_gii, rh_gii, bids, aid]
                }
            def t1w_vol_only = tsnr_t1w_no_surf
                .map { sub, ses, avg, bids, space, aid -> [sub, ses, avg, dummy_sf_lh, dummy_sf_rh, bids, ''] }
            def other_vol_only = tsnr_other_ch
                .map { sub, ses, avg, bids, space -> [sub, ses, avg, dummy_sf_lh, dummy_sf_rh, bids, ''] }

            tsnr_qc_input = t1w_with_surf.mix(t1w_vol_only).mix(other_vol_only)
        } else {
            tsnr_qc_input = FUNC_TSNR_SESSION_AVERAGE.out.output
                .map { sub, ses, avg, bids, space -> [sub, ses, avg, dummy_sf_lh, dummy_sf_rh, bids, ''] }
        }

        QC_TSNR(tsnr_qc_input, config_file)

        // ============================================
        // CONFOUNDS (fMRIPrep-compatible)
        // ============================================
        // Gated by func.confounds.enabled (default on). The confounds TSV is a derivative only
        // (the BOLD image is never modified). Motion-derived columns (24-param, framewise_displacement,
        // rmsd) are present only when motion correction ran; the remaining regressors (dvars/std_dvars,
        // global_signal, tissue, DVARS outliers) are computed regardless, so confounds stay useful even
        // with motion correction disabled.
        if (func_confounds_enabled) {
            // motion (24-param) + framewise_displacement + rmsd + dvars/std_dvars + global_signal +
            // outliers, computed in the registered (preferably space-T1w) BOLD space. When a T1w
            // anatomical segmentation is available, csf/white_matter/csf_wm tissue regressors are added too.
            //
            // Per-session segmentation + LUT, keyed to each func session via the matched anatomical
            // session (`anat_ses` from func_anat_selection). anat_seg_lut covers only sessions with a real
            // segmentation; we then make it total over all func sessions with dummy sentinels.
            //
            // Session keys must be normalized before matching: func_anat_selection emits anat_ses="" for
            // subject-level anatomy while anat_skull_seg carries the native session (null). So we combine
            // by subject only and filter on normalized-session equality (same idiom as the registration
            // wiring above), instead of an exact combine(by:[0,1]) that would treat ""!=null and silently
            // drop tissue regressors for subjects with subject-level anatomy.
            def normSes = channelHelpers.normalizeSessionId
            def anat_seg_lut = anat_skull_seg.join(anat_skull_seg_lut, by: [0, 1])  // [sub, anat_seg_ses, seg, lut]
            def func_seg_lut_real = func_anat_selection
                .map { sub, ses_func, anat_file, anat_ses -> [sub, ses_func, anat_ses] }
                .combine(anat_seg_lut, by: 0)  // by subject: [sub, ses_func, anat_ses, anat_seg_ses, seg, lut]
                .filter { sub, ses_func, anat_ses, anat_seg_ses, seg, lut -> normSes(anat_ses) == normSes(anat_seg_ses) }
                .map { sub, ses_func, anat_ses, anat_seg_ses, seg, lut -> [sub, ses_func, seg, lut] }
                .unique()
            def dummy_seg = file("${workDir}/dummy_confounds_seg.dummy").tap { it.toFile().text = "" }
            def dummy_lut = file("${workDir}/dummy_confounds_lut.dummy").tap { it.toFile().text = "" }
            def dummy_motion = file("${workDir}/dummy_confounds_motion.dummy").tap { it.toFile().text = "" }
            def func_seg_lut = func_anat_selection
                .map { sub, ses_func, anat_file, anat_ses -> [sub, ses_func] }
                .unique()
                .join(func_seg_lut_real, by: [0, 1], remainder: true)
                .map { row ->
                    def sub = row[0]; def ses = row[1]
                    def seg = (row.size() > 2 && row[2]) ? row[2] : dummy_seg
                    def lut = (row.size() > 3 && row[3]) ? row[3] : dummy_lut
                    [sub, ses, seg, lut]
                }

            // Make motion params total over every TSNR run: drive a remainder join from the run keys so
            // runs with no motion TSV (motion correction disabled -> func_motion_params is empty) get a
            // .dummy sentinel that FUNC_COMPUTE_CONFOUNDS' _real() maps back to None (Python then omits
            // the motion columns). Building it run-key-driven keeps this total, so the join below stays a
            // plain inner join without ever dropping a run.
            def motion_for_runs = tsnr_run_input
                .map { sub, ses, run_id, bold_sel, bids_name, mask_sel, space_label -> [sub, ses, run_id] }
                .unique()
                .join(func_motion_params.map { sub, ses, run_id, motion_tsv -> [sub, ses, run_id, motion_tsv] }, by: [0, 1, 2], remainder: true)
                .map { row -> [row[0], row[1], row[2], (row.size() > 3 && row[3]) ? row[3] : dummy_motion] }

            // Reuse the TSNR bold+mask channel [sub,ses,run, bold, bids, mask, space], inner-join the
            // per-run motion params (rmsd is derived in Python so no separate rel-RMS channel), then
            // attach the session's seg+lut — gated to T1w-space runs since the seg is T1w-space.
            def confounds_input = tsnr_run_input
                .join(motion_for_runs, by: [0, 1, 2])
                .map { sub, ses, run_id, bold_sel, bids_name, mask_sel, space_label, motion_tsv ->
                    [sub, ses, run_id, bold_sel, mask_sel, motion_tsv, bids_name, space_label]
                }
                .combine(func_seg_lut, by: [0, 1])
                .map { sub, ses, run_id, bold_sel, mask_sel, motion_tsv, bids_name, space_label, seg, lut ->
                    def use_seg = (space_label == 'T1w') ? seg : dummy_seg
                    def use_lut = (space_label == 'T1w') ? lut : dummy_lut
                    [sub, ses, run_id, bold_sel, mask_sel, motion_tsv, bids_name, use_seg, use_lut]
                }

            FUNC_COMPUTE_CONFOUNDS(confounds_input, config_file)

            // QC: fMRIPrep-style confound line panels (reads only the confounds TSV).
            // FUNC_COMPUTE_CONFOUNDS.out.output = [sub, ses, run_id, confounds_tsv, bids_name]
            QC_CONFOUNDS(FUNC_COMPUTE_CONFOUNDS.out.output, config_file)

            // Same TSV feeds the motion QC figure so both stacked plots shade the same frames.
            confounds_for_qc = FUNC_COMPUTE_CONFOUNDS.out.output
        }
    }

    // ============================================
    // QUALITY CONTROL
    // ============================================
    // Generate QC reports for functional processing steps
    if (func_motion_correction_enabled) {
        func_after_motion
            .join(func_motion_params, by: [0, 1, 2])
            .map { sub, ses, run_identifier, bold_file, tmean_file, bids_name, motion_file ->
                [sub, ses, run_identifier, motion_file, tmean_file, bids_name]
            }
            .set { motion_qc_base }

        // Make the confounds TSV total over the motion-QC run keys: drive a remainder join from
        // those keys so runs with no confounds (registration/confounds disabled, single-volume runs
        // skipped by FUNC_COMPUTE_CONFOUNDS, or a run that errored out) get a .dummy sentinel that
        // QC_MOTION_CORRECTION's _real() maps back to None. Keeping it run-key-driven means the
        // join below never drops a run from motion QC. (Same idiom as motion_for_runs above.)
        def dummy_motion_qc_confounds = file("${workDir}/dummy_motion_qc_confounds.dummy").tap { it.toFile().text = "" }
        def confounds_for_motion_qc = motion_qc_base
            .map { sub, ses, run_id, motion_file, tmean_file, bids_name -> [sub, ses, run_id] }
            .unique()
            .join(
                confounds_for_qc.map { sub, ses, run_id, confounds_tsv, bids_name -> [sub, ses, run_id, confounds_tsv] },
                by: [0, 1, 2],
                remainder: true
            )
            .map { row -> [row[0], row[1], row[2], (row.size() > 3 && row[3]) ? row[3] : dummy_motion_qc_confounds] }

        motion_qc_base
            .join(confounds_for_motion_qc, by: [0, 1, 2])
            .set { motion_qc_input }
        QC_MOTION_CORRECTION(motion_qc_input, config_file)
    }
    
    if (func_conform_enabled) {
        // Join conform output with reference
        // func_apply_conform_output: [sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name]
        // FUNC_COMPUTE_CONFORM.out.reference: [sub, ses, run_id, reference.nii.gz]
        // When coreg is enabled, compute phase is session-level (run_id == ""), so combine by [0, 1] instead of [0, 1, 2]
        def func_conform_qc_input = isSessionLevel ?
            func_apply_conform_output
                .combine(FUNC_COMPUTE_CONFORM.out.reference, by: [0, 1])
                .map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name, run_id2, reference ->
                    // QC expects: [sub, ses, run_id, conformed_file, bids_name, reference_file]
                    [sub, ses, run_id, conformed_tmean_ref, bids_name, reference]
                } :
            func_apply_conform_output
                .join(FUNC_COMPUTE_CONFORM.out.reference, by: [0, 1, 2])
                .map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name, reference ->
                    // QC expects: [sub, ses, run_id, conformed_file, bids_name, reference_file]
                    [sub, ses, run_id, conformed_tmean_ref, bids_name, reference]
                }
        QC_CONFORM_FUNC(func_conform_qc_input, config_file)
    }
    
    if (func_skullstripping_enabled) {
        // Session-level compute phase: FUNC_COMPUTE_BRAIN_MASK.out.output has run_id == ""
        // Per-run compute phase: FUNC_COMPUTE_BRAIN_MASK.out.output has run-level items
        def func_skull_qc_input = isSessionLevel ?
            func_apply_conform_output
                .map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name ->
                    [sub, ses, run_id, conformed_tmean_ref, bids_name]
                }
                .combine(
                    FUNC_COMPUTE_BRAIN_MASK.out.output
                        .map { sub, ses, run_id, masked_tmean, bids_name, brain_mask ->
                            [sub, ses, brain_mask]
                        },
                    by: [0, 1]
                )
                .map { sub, ses, run_id, conformed_tmean_ref, bids_name, brain_mask ->
                    [sub, ses, run_id, conformed_tmean_ref, brain_mask, bids_name]
                } :
            func_apply_conform_output
                .map { sub, ses, run_id, conformed_bold, conformed_tmean_ref, bids_name ->
                    [sub, ses, run_id, conformed_tmean_ref, bids_name]
                }
                .join(
                    FUNC_COMPUTE_BRAIN_MASK.out.output
                        .map { sub, ses, run_id, masked_tmean, bids_name, brain_mask ->
                            [sub, ses, run_id, brain_mask]
                        },
                    by: [0, 1, 2]
                )
                .map { sub, ses, run_id, conformed_tmean_ref, bids_name, brain_mask ->
                    [sub, ses, run_id, conformed_tmean_ref, brain_mask, bids_name]
                }
        QC_SKULLSTRIPPING_FUNC(func_skull_qc_input, config_file)
    }

    if (registration_enabled) {
        // QC for intermediate func2anat step (only when sequential transforms are used)
        func_apply_reg_intermediate
            .map { sub, ses, run_identifier, intermediate_boldref, intermediate_reference, bids_name ->
                [sub, ses, run_identifier, intermediate_boldref, intermediate_reference, bids_name]
            }
            .set { func_reg_intermediate_qc_input }
        QC_REGISTRATION_FUNC_INTERMEDIATE(func_reg_intermediate_qc_input, "func2anat", config_file)
        
        // QC for final func2target step (template space)
        func_apply_reg
            .join(func_apply_reg_reference, by: [0, 1, 2])
            .map { sub, ses, run_identifier, registered_bold, registered_boldref, bids_name, reference_file ->
                [sub, ses, run_identifier, registered_boldref, reference_file, bids_name]
            }
            .set { func_reg_qc_input }
        QC_REGISTRATION_FUNC(func_reg_qc_input, "func2target", config_file)
    }
    
    // ============================================
    // COLLECT QC CHANNELS
    // ============================================
    // Aggregate all QC metadata channels for output
    func_qc_channels = Channel.empty()
    func_jobs_ch_out = Channel.empty()
    
    func_jobs_ch_out = func_jobs_ch
    if (func_motion_correction_enabled) {
        func_qc_channels = func_qc_channels.mix(QC_MOTION_CORRECTION.out.metadata)
    }
    if (func_conform_enabled) {
        func_qc_channels = func_qc_channels.mix(QC_CONFORM_FUNC.out.metadata)
    }
    if (func_skullstripping_enabled) {
        func_qc_channels = func_qc_channels.mix(QC_SKULLSTRIPPING_FUNC.out.metadata)
    }
    if (func_coreg_runs_within_session) {
        func_qc_channels = func_qc_channels.mix(QC_WITHIN_SES_COREG.out.metadata)
    }
    if (registration_enabled) {
        func_qc_channels = func_qc_channels.mix(QC_REGISTRATION_FUNC_INTERMEDIATE.out.metadata)
        func_qc_channels = func_qc_channels.mix(QC_REGISTRATION_FUNC.out.metadata)
        func_qc_channels = func_qc_channels.mix(QC_TSNR.out.metadata)
    }

    emit:
    func_qc_channels
    func_jobs_ch_out
}
