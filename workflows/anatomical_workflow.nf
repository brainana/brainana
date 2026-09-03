/*
 * Anatomical Processing Workflow
 *
 * Handles all anatomical processing steps including:
 * - Anatomical job parsing
 * - Anatomical processing pipeline
 * - T2w special processing
 * - Anatomical QC steps
 *
 * Surface reconstruction is in surfrecon_workflow.nf (SURF_RECON_WF).
 */

nextflow.enable.dsl=2

// Include anatomical processing modules
include { ANAT_SYNTHESIS } from '../modules/anatomical.nf'
include { ANAT_SYNTHESIS as ANAT_SYNTHESIS_T2W } from '../modules/anatomical.nf'
include { ANAT_CONFORM } from '../modules/anatomical.nf'
include { ANAT_CONFORM as ANAT_CONFORM_T2W } from '../modules/anatomical.nf'
include { ANAT_BIAS_CORRECTION } from '../modules/anatomical.nf'
include { ANAT_BIAS_CORRECTION as ANAT_BIAS_CORRECTION_T2W } from '../modules/anatomical.nf'
include { ANAT_SKULLSTRIPPING } from '../modules/anatomical.nf'
include { ANAT_SKULLSTRIPPING as ANAT_SKULLSTRIPPING_T2W } from '../modules/anatomical.nf'
include { ANAT_REGISTRATION } from '../modules/anatomical.nf'
include { ANAT_REGISTRATION as ANAT_REGISTRATION_T2W } from '../modules/anatomical.nf'
include { ANAT_T2W_TO_T1W_REGISTRATION } from '../modules/anatomical.nf'
include { ANAT_CONFORM_PASSTHROUGH } from '../modules/anatomical.nf'
include { ANAT_CONFORM_PASSTHROUGH as ANAT_CONFORM_PASSTHROUGH_T2W } from '../modules/anatomical.nf'
include { ANAT_BIAS_CORRECTION_PASSTHROUGH } from '../modules/anatomical.nf'
include { ANAT_BIAS_CORRECTION_PASSTHROUGH as ANAT_BIAS_CORRECTION_PASSTHROUGH_T2W } from '../modules/anatomical.nf'
include { ANAT_REGISTRATION_PASSTHROUGH } from '../modules/anatomical.nf'
include { ANAT_REGISTRATION_PASSTHROUGH as ANAT_REGISTRATION_PASSTHROUGH_T2W } from '../modules/anatomical.nf'
include { ANAT_APPLY_CONFORM } from '../modules/anatomical.nf'
include { ANAT_APPLY_TRANSFORMATION } from '../modules/anatomical.nf'
include { ANAT_APPLY_TRANSFORM_MASK } from '../modules/anatomical.nf'
include { ANAT_BACKPROJECT_ATLASES_TO_T1W } from '../modules/anatomical.nf'
include { ANAT_BACKPROJECT_ATLASES_TO_SCANNER } from '../modules/anatomical.nf'
include { ANAT_PROJECT_ATLASES_TO_SURFACE } from '../modules/anatomical.nf'
include { ANAT_PUBLISH_PHASE1 } from '../modules/anatomical.nf'
include { ANAT_PUBLISH_PHASE1 as ANAT_PUBLISH_T2W } from '../modules/anatomical.nf'
include { ANAT_T1WT2W_COMBINED } from '../modules/anatomical.nf'

// Include anatomical QC modules
include { QC_CONFORM } from '../modules/qc.nf'
include { QC_BIAS_CORRECTION } from '../modules/qc.nf'
include { QC_BIAS_CORRECTION as QC_BIAS_CORRECTION_T2W } from '../modules/qc.nf'
include { QC_SKULLSTRIPPING } from '../modules/qc.nf'
include { QC_ATLAS_SEGMENTATION } from '../modules/qc.nf'
include { QC_REGISTRATION } from '../modules/qc.nf'
include { QC_T2W_TO_T1W_REGISTRATION } from '../modules/qc.nf'
include { QC_T2W_TEMPLATE_SPACE } from '../modules/qc.nf'
include { QC_T1WT2W_COMBINED } from '../modules/qc.nf'
include { SURF_RECON_WF } from './surfrecon_workflow.nf'

// Load external Groovy files for channel operations
def channelHelpers = evaluate(new File("${projectDir}/workflows/channel_helpers.groovy").text)

// Load parameter resolver
def paramResolver = evaluate(new File("${projectDir}/workflows/param_resolver.groovy").text)

// Load config helpers
def configHelpers = evaluate(new File("${projectDir}/workflows/config_helpers.groovy").text)

workflow ANAT_WF {
    take:
    gpu_queue
    
    main:
    // ============================================
    // INPUT VALIDATION
    // ============================================
    if (!params.bids_dir) {
        error "Missing required parameter: --bids_dir"
    }
    if (!params.output_dir) {
        error "Missing required parameter: --output_dir"
    }
    
    // Validate paths exist
    def bids_dir_path = file(params.bids_dir)
    def bids_dir_str = params.bids_dir.toString()
    if (!new File(bids_dir_str).exists()) {
        error "BIDS directory not found: ${params.bids_dir}"
    }
    
    // ============================================
    // INITIALIZATION
    // ============================================
    // Initialize parameter resolver (if not already initialized in main.nf)
    configHelpers.ensureParamResolverInitialized(paramResolver, params, projectDir)
    
    // Use effective config file (generated in main.nf)
    def config_file_path = configHelpers.getEffectiveConfigPath(params, projectDir)
    def config_file = config_file_path
    
    // ============================================
    // RESOLVE PARAMETERS (for workflow logic)
    // ============================================
    // Priority: CLI params → YAML config → defaults.yaml
    // All defaults come from defaults.yaml - no hardcoded values
    def use_t1wt2wcombined = paramResolver.getYamlBool("anat.surface_reconstruction.use_t1wt2wcombined")
    def anat_conform_enabled = paramResolver.getYamlBool("anat.conform.enabled")
    def anat_bias_correction_enabled = paramResolver.getYamlBool("anat.bias_correction.enabled")
    def anat_skullstripping_enabled = paramResolver.getYamlBool("anat.skullstripping_segmentation.enabled")
    def registration_enabled = paramResolver.getYamlBool("registration.enabled", true)
    
    // Resolve BIDS filtering parameters (CLI parameters)
    // Defaults come from defaults.yaml (bids_filtering.*)
    def subjects_list = paramResolver.getParamList(params, 'subjects', null)
    def sessions_list = paramResolver.getParamList(params, 'sessions', null)
    def tasks_list = paramResolver.getParamList(params, 'tasks', null)
    def runs_list = paramResolver.getParamList(params, 'runs', null)
    
    // Convert lists to strings for display (backward compatibility with existing code)
    def subjects_str = subjects_list ? subjects_list.join(' ') : ''
    def sessions_str = sessions_list ? sessions_list.join(' ') : ''
    def tasks_str = tasks_list ? tasks_list.join(' ') : ''
    def runs_str = runs_list ? runs_list.join(' ') : ''
    
    // Get output_space for display (from effective config)
    def effective_output_space = paramResolver.getParamOutputSpace(params, 'output_space')
    // Custom templates carry no bundled atlases, so atlas back-/surface-projection has
    // nothing to do. Decide this once here (a run-level fact) and gate the surface step below.
    def is_custom_template = paramResolver.isCustomTemplatePath(effective_output_space)
    
    println "============================================"
    println "brainana Nextflow Pipeline - Anatomical"
    println "============================================"
    println "BIDS directory: ${params.bids_dir}"
    println "Output directory: ${params.output_dir}"
    println "Output space: ${effective_output_space}"
    println "Effective config: ${config_file_path}"
    if (subjects_str) println "Subjects filter: ${subjects_str}"
    if (sessions_str) println "Sessions filter: ${sessions_str}"
    if (tasks_str) println "Tasks filter: ${tasks_str}"
    if (runs_str) println "Runs filter: ${runs_str}"
    println "============================================"
    
    // ============================================
    // PARSE ANATOMICAL JOBS
    // ============================================
    // Load pre-discovered job lists from discovery script
    // Channel structure: [sub, ses, file_objects, needs_synth, suffix, needs_t1w_reg]
    def anat_jobs_file = file("${params.output_dir}/nextflow_reports/anatomical_jobs.json")
    
    if (!new File(anat_jobs_file.toString()).exists()) {
        error "Discovery file not found: ${anat_jobs_file}\n" +
              "Please run the discovery script before starting Nextflow."
    }
    
    Channel.fromPath(anat_jobs_file)
        .splitJson()
        .map { job ->
            def sub = job.subject_id.toString()
            def ses = job.session_id ? job.session_id.toString() : null
            def needs_synth = job.needs_synthesis ?: false
            def file_paths_raw = job.file_paths ?: [job.file_path]
            def file_paths_list = file_paths_raw instanceof List ? file_paths_raw : [file_paths_raw]
            def file_objects = file_paths_list.collect { path_val ->
                file(path_val as String)
            }
            def suffix = job.suffix.toString()
            def needs_t1w_reg = job.needs_t1w_registration ?: false
            def synthesis_type = job.synthesis_type ?: null
            
            [sub, ses, file_objects, needs_synth, suffix, needs_t1w_reg, synthesis_type]
        }
        .set { anat_jobs_ch }
    
    // ============================================
    // PREPARE CHANNEL HELPERS
    // ============================================
    def getSingleFilePath = channelHelpers.getSingleFilePath
    def isT1wFile = channelHelpers.isT1wFile
    def passThroughAnat = channelHelpers.passThroughAnat
    def normalizeSessionId = channelHelpers.normalizeSessionId
    def matchSessions = channelHelpers.matchSessions
    
    // Extract all subjects from anat_jobs_ch for later use
    anat_jobs_ch
        .map { it[0] }
        .unique()
        .set { anat_subjects_ch }
    
    // ============================================
    // BRANCH JOBS BY TYPE
    // ============================================
    // T1w vs T2w only. All jobs run through ANAT_SYNTHESIS / ANAT_SYNTHESIS_T2W so Python
    // can passthrough single files (n<=1) while applying consistent space-scanner naming.
    // needs_synthesis on the job record is metadata only for reporting.
    // Input: anat_jobs_ch: [sub, ses, file_objects, needs_synth, suffix, needs_t1w_reg, synthesis_type]
    // ============================================
    anat_jobs_ch.branch {
        t1w: it[4] == "T1w"
        t2w: it[4] == "T2w"
    }.set { anat_branched }
    
    // ============================================
    // ALL T1W JOBS → ANAT_SYNTHESIS
    // ============================================
    // Input: anat_branched.t1w: [sub, ses, file_objects, needs_synth, suffix, needs_t1w_reg, synthesis_type]
    // Output: anat_input_ch: [sub, ses, anat_file, bids_name]
    // ============================================
    def anat_t1w_input = anat_branched.t1w
        .map { item ->
            def sub = item[0]
            def ses = item[1]
            def file_objects = item[2]
            [sub, ses, file_objects]
        }
    
    ANAT_SYNTHESIS(anat_t1w_input, config_file)
    def anat_input_ch = ANAT_SYNTHESIS.out.synthesized
        .map { sub, ses, anat_file, bids_name_file ->
            def bids_name = bids_name_file.text.trim()
            [sub, ses, anat_file, bids_name]
        }
    
    // ============================================
    // ALL T2W JOBS → ANAT_SYNTHESIS_T2W
    // ============================================
    // Fork t2w branch (single consumer per channel): process inputs + needs_t1w_reg for downstream join.
    // Output: anat_t2w_all_jobs: [sub, ses, anat_file, bids_name, needs_t1w_reg]
    // ============================================
    anat_branched.t2w.multiMap { item ->
        def sub = item[0]
        def ses = item[1]
        def file_objects = item[2]
        def needs_t1w_reg = item[5]
        for_synthesis: [sub, ses, file_objects]
        needs_t1w_reg_lookup: [sub, ses, needs_t1w_reg]
    }.set { t2w_synthesis_fork }
    
    ANAT_SYNTHESIS_T2W(t2w_synthesis_fork.for_synthesis, config_file)
    def anat_t2w_all_jobs = ANAT_SYNTHESIS_T2W.out.synthesized
        .map { sub, ses, anat_file, bids_name_file ->
            def bids_name = bids_name_file.text.trim()
            [sub, ses, anat_file, bids_name]
        }
        .join(t2w_synthesis_fork.needs_t1w_reg_lookup, by: [0, 1])
        .map { sub, ses, anat_file, bids_name, needs_t1w_reg ->
            [sub, ses, anat_file, bids_name, needs_t1w_reg]
        }
    
    // ============================================
    // ANATOMICAL PROCESSING PIPELINE
    // ============================================
    // Sequential processing: conform → skull stripping → bias correction → registration
    // Channel structure maintained: [sub, ses, anat_file, bids_name]
    // ============================================
    
    // ============================================
    // Use synthesis output directly before conform.
    anat_preconform = anat_input_ch.map(passThroughAnat)
    
    // ============================================
    // CONFORM
    // ============================================
    // Conform anatomical images to template space
    // Input: anat_preconform: [sub, ses, anat_file, bids_name]
    // Output: anat_after_conform: [sub, ses, anat_file, bids_name]
    //         anat_conform_transforms: [sub, ses, forward_xfm, inverse_xfm]
    //         anat_conform_reference: [sub, ses, reference_file]
    // ============================================
    anat_after_conform = anat_preconform
    anat_conform_transforms = Channel.empty()
    anat_conform_reference = Channel.empty()
    // Full-FOV conform: [sub, ses, uncropped_image, uncropped_template, fov_box].
    // Leaf product - QC underlay only, no processing step reads it.
    anat_conform_full_fov = Channel.empty()
    if (anat_conform_enabled) {
        ANAT_CONFORM(anat_preconform, config_file)
        anat_after_conform = ANAT_CONFORM.out.output
        anat_conform_transforms = ANAT_CONFORM.out.transforms
        anat_conform_reference = ANAT_CONFORM.out.reference
        anat_conform_full_fov = ANAT_CONFORM.out.full_fov
    } else {
        ANAT_CONFORM_PASSTHROUGH(anat_preconform, config_file)
        anat_after_conform = ANAT_CONFORM_PASSTHROUGH.out.output
        anat_conform_transforms = ANAT_CONFORM_PASSTHROUGH.out.transforms
        anat_conform_reference = ANAT_CONFORM_PASSTHROUGH.out.reference
    }
    
    // ============================================
    // SKULLSTRIPPING
    // ============================================
    // Remove non-brain tissue using deep learning segmentation
    // Input: anat_after_conform: [sub, ses, anat_file, bids_name]
    // Output: anat_after_skull: [sub, ses, anat_file, bids_name] (full head, always - not skullstripped)
    //         anat_after_skull_brain: [sub, ses, brain_file, bids_name] (brain-only, always available - real or dummy)
    //         anat_skull_mask: [sub, ses, brain_mask]
    //         anat_skull_seg: [sub, ses, brain_segmentation]
    // Principle: anat_after_xxxstep = full head (_T1w), anat_after_xxxstep_brain = brain (_T1w_brain)
    // ============================================
    // Initialize with dummy mask and segmentation from anat_after_conform (never null)
    def dummy_mask = file("${workDir}/dummy_brain_mask.dummy")
    dummy_mask.toFile().text = ""
    def dummy_seg = file("${workDir}/dummy_brain_segmentation.dummy")
    dummy_seg.toFile().text = ""
    def dummy_brain = file("${workDir}/dummy_brain.dummy")
    dummy_brain.toFile().text = ""
    def anat_skull_mask_dummy = anat_after_conform.map { sub, ses, anat_file, bids_name ->
        [sub, ses, dummy_mask]
    }
    def anat_skull_seg_dummy = anat_after_conform.map { sub, ses, anat_file, bids_name ->
        [sub, ses, dummy_seg]
    }
    def anat_after_skull_dummy = anat_after_conform.map(passThroughAnat)
    def anat_after_skull_brain_dummy = anat_after_conform.map { sub, ses, anat_file, bids_name ->
        [sub, ses, dummy_brain, bids_name]
    }
    
    // Always initialize with dummy segmentation to ensure consistent structure for joins/combines.
    // No `def`: workflow-scoped so it can be emitted (see surf_actual_subject_id below).
    anat_skull_seg = anat_skull_seg_dummy
    def anat_after_skull = anat_after_skull_dummy
    def anat_after_skull_brain = anat_after_skull_brain_dummy
    
    // Always initialize with dummy mask to ensure valid structure for joins/combines
    // This ensures Nextflow can validate structure at parse time
    def anat_skull_mask = anat_skull_mask_dummy
    
    // Atlas LUT (optional): only when skullstripping + multi-class atlas; empty when disabled.
    // No `def`: workflow-scoped so it can be emitted.
    anat_skull_seg_lut = Channel.empty()
    
    if (anat_skullstripping_enabled) {
        // Use GPU token only when workflow-level GPU scheduling is enabled (use_gpu).
        // Without this gate, skullstripping always pulls gpu_id=0 from gpu_queue and runs on
        // GPU even in CPU mode (general.gpu_device=-1 / use_gpu=false).
        def use_skull_gpu = params.use_gpu
        def skull_gpu_input = use_skull_gpu ? gpu_queue : Channel.value('none')
        ANAT_SKULLSTRIPPING(anat_after_conform, config_file, skull_gpu_input)
        if (use_skull_gpu) {
            ANAT_SKULLSTRIPPING.out.gpu_token.subscribe { gpu_queue << it }
        }
        // Principle: anat_after_skull = full head (not skullstripped), anat_after_skull_brain = brain (skullstripped)
        anat_after_skull = ANAT_SKULLSTRIPPING.out.output  // Full head version (_T1w)
        anat_after_skull_brain = ANAT_SKULLSTRIPPING.out.brain  // Brain-only version (_T1w_brain)
        // Create a channel that prefers process output but falls back to dummy
        // Use groupTuple to handle cases where both exist, then prefer non-dummy
        def skull_mask_with_fallback = ANAT_SKULLSTRIPPING.out.brain_mask
            .mix(anat_skull_mask_dummy)
            .groupTuple(by: [0, 1])
            .map { sub, ses, mask_list ->
                // Prefer real mask (non-dummy) if available, otherwise use first item
                def real_mask = mask_list.find { mask -> !mask.toString().contains('.dummy') }
                [sub, ses, real_mask ?: mask_list[0]]
            }
        anat_skull_mask = skull_mask_with_fallback
        // Use real segmentation when available, otherwise keep dummy
        def skull_seg_with_fallback = ANAT_SKULLSTRIPPING.out.brain_segmentation
            .mix(anat_skull_seg_dummy)
            .groupTuple(by: [0, 1])
            .map { sub, ses, seg_list ->
                // Prefer real segmentation (non-dummy) if available, otherwise use first item
                def real_seg = seg_list.find { seg -> !seg.toString().contains('.dummy') }
                [sub, ses, real_seg ?: seg_list[0]]
            }
        anat_skull_seg = skull_seg_with_fallback
        anat_skull_seg_lut = ANAT_SKULLSTRIPPING.out.brain_segmentation_lut
    }
    
    // ============================================
    // BIAS CORRECTION
    // ============================================
    // Correct intensity non-uniformity (bias field) using brain mask
    // Input: anat_after_conform: [sub, ses, anat_file, bids_name] (full head T1w)
    //        anat_skull_mask: [sub, ses, brain_mask] (may be dummy)
    // Output: anat_after_bias: [sub, ses, anat_file, bids_name] (bias-corrected full head _T1w)
    //         anat_after_bias_brain: [sub, ses, brain_file, bids_name] (bias-corrected brain _T1w_brain, always available - real or dummy)
    // Principle: anat_after_xxxstep = full head (_T1w), anat_after_xxxstep_brain = brain (_T1w_brain)
    // ============================================
    anat_after_bias = anat_after_conform.map(passThroughAnat)
    anat_after_bias_brain = Channel.empty()
    if (anat_bias_correction_enabled) {
        // Join conformed T1w with mask (both have [sub, ses] as first two elements)
        // Now that structures are consistent with dummies, we can use join() for exact matching
        def bias_correction_input = anat_after_conform
            .join(anat_skull_mask, by: [0, 1])
        
        // Split into cases with real mask vs dummy mask
        def bias_correction_with_mask = bias_correction_input
            .filter { sub, ses, anat_file, bids_name, mask_file ->
                !mask_file.toString().contains('.dummy')
            }
        
        def bias_correction_no_mask = bias_correction_input
            .filter { sub, ses, anat_file, bids_name, mask_file ->
                mask_file.toString().contains('.dummy')
            }
            .map { sub, ses, anat_file, bids_name, mask_file ->
                [sub, ses, anat_file, bids_name]
            }
        
        // Prepare input channels for bias correction with real mask
        def bias_input_files = bias_correction_with_mask
            .map { sub, ses, anat_file, bids_name, mask_file ->
                [sub, ses, anat_file, bids_name]
            }
        def bias_input_masks = bias_correction_with_mask
            .map { sub, ses, anat_file, bids_name, mask_file ->
                [sub, ses, mask_file]
            }
        
        // Extract bids_name lookup for with_mask path (needed for joining brain output)
        // Create this BEFORE consuming bias_input_files in the process call
        def bias_with_mask_bids_lookup = bias_correction_with_mask
            .map { sub, ses, anat_file, bids_name, mask_file ->
                [sub, ses, bids_name]
            }
        
        // Extract bids_name lookup for no_mask path (needed for joining brain output)
        def bias_no_mask_bids_lookup = bias_correction_no_mask
            .map { sub, ses, anat_file, bids_name ->
                [sub, ses, bids_name]
            }
        
        // Run bias correction with mask (process will only run if channel has items)
        ANAT_BIAS_CORRECTION(bias_input_files, bias_input_masks, config_file)
        def bias_with_mask_output = ANAT_BIAS_CORRECTION.out.output
        // Join brain output with bids_name from lookup to match anat_after_skull structure [sub, ses, brain_file, bids_name]
        def bias_with_mask_brain = ANAT_BIAS_CORRECTION.out.brain
            .join(bias_with_mask_bids_lookup, by: [0, 1])
            .map { sub, ses, brain_file, bids_name ->
                [sub, ses, brain_file, bids_name]
            }
        
        // Use passthrough for cases with dummy mask (always outputs dummy brain)
        ANAT_BIAS_CORRECTION_PASSTHROUGH(bias_correction_no_mask, config_file)
        def bias_no_mask_output = ANAT_BIAS_CORRECTION_PASSTHROUGH.out.output
        // Join brain output with bids_name to match anat_after_skull structure [sub, ses, brain_file, bids_name]
        def bias_no_mask_brain = ANAT_BIAS_CORRECTION_PASSTHROUGH.out.brain
            .join(bias_no_mask_bids_lookup, by: [0, 1])
            .map { sub, ses, brain_file, bids_name ->
                [sub, ses, brain_file, bids_name]
            }
        
        // Mix outputs from both paths
        // Since both processes always output brain (real or dummy), and we join with bias_bids_lookup
        // (which always has entries), both brain channels should have entries after join
        anat_after_bias = bias_with_mask_output.mix(bias_no_mask_output)
        anat_after_bias_brain = bias_with_mask_brain.mix(bias_no_mask_brain)
    } else {
        // If bias correction is disabled, use passthrough to maintain channel structure
        // Passthrough always outputs dummy brain for consistent structure
        // Extract bids_name lookup BEFORE consuming anat_after_conform in process call
        def bias_passthrough_bids_lookup = anat_after_conform
            .map { sub, ses, anat_file, bids_name -> [sub, ses, bids_name] }
        
        ANAT_BIAS_CORRECTION_PASSTHROUGH(anat_after_conform, config_file)
        anat_after_bias = ANAT_BIAS_CORRECTION_PASSTHROUGH.out.output
        // Join brain output with bids_template to match anat_after_skull structure [sub, ses, brain_file, bids_name]
        anat_after_bias_brain = ANAT_BIAS_CORRECTION_PASSTHROUGH.out.brain
            .join(bias_passthrough_bids_lookup, by: [0, 1])
            .map { sub, ses, brain_file, bids_name ->
                [sub, ses, brain_file, bids_name]
            }
    }
    
    // ============================================
    // PUBLISH PHASE 1 OUTPUTS
    // ============================================
    // Publish Phase 1 preprocessed outputs (desc-preproc naming)
    // Takes final Phase 1 outputs and creates desc-preproc versions for publishing
    // Input: anat_after_bias: [sub, ses, anat_file, bids_name]
    //        anat_after_bias_brain: [sub, ses, brain_file, bids_name]
    // Output: Creates desc-preproc symlinks and publishes them
    // ============================================
    def phase1_publish_input = anat_after_bias
        .join(anat_after_bias_brain, by: [0, 1])
        .map { sub, ses, anat_file, anat_bids, brain_file, brain_bids ->
            [sub, ses, anat_file, brain_file, anat_bids]
        }
    
    ANAT_PUBLISH_PHASE1(phase1_publish_input, config_file)
    
    // ============================================
    // REGISTRATION
    // ============================================
    // Register anatomical images to template space
    // Compute transform on bias-corrected brain version (for better registration)
    // Apply transform to unskullstripped version (to get full head in template space)
    // Input: anat_after_bias_brain: [sub, ses, brain_file, bids_name] (for computing transform, always available - real or dummy)
    //        anat_after_bias: [sub, ses, anat_file, bids_name] (for applying transform - bias-corrected full head _T1w)
    // Output: anat_after_reg: [sub, ses, registered_file, bids_name] (full head registered _T1w, not brain)
    //         anat_reg_transforms: [sub, ses, bids_name, forward_xfm, inverse_xfm]
    //         anat_reg_reference: [sub, ses, reference_file]
    // Principle: anat_after_xxxstep = full head (_T1w), anat_after_xxxstep_brain = brain (_T1w_brain)
    // ============================================
    anat_after_reg = Channel.empty()
    anat_reg_transforms = Channel.empty()
    anat_reg_reference = Channel.empty()
    if (registration_enabled) {
        // Join brain version (for computing transform) with bias-corrected full head (for applying transform)
        def registration_input = anat_after_bias_brain
            .join(anat_after_bias, by: [0, 1])
            .map { sub, ses, brain_file, brain_bids_name, anat_file, anat_bids_name ->
                [sub, ses, brain_file, brain_bids_name, anat_file]
            }
        
        // Use GPU token only when workflow-level GPU scheduling is enabled.
        def use_registration_gpu = params.use_gpu
        def gpu_input = use_registration_gpu ? gpu_queue : Channel.value('none')
        
        ANAT_REGISTRATION(registration_input, config_file, gpu_input)
        if (use_registration_gpu) {
            ANAT_REGISTRATION.out.gpu_token.subscribe { gpu_queue << it }
        }
        anat_after_reg = ANAT_REGISTRATION.out.output
        anat_reg_transforms = ANAT_REGISTRATION.out.transforms
        anat_reg_reference = ANAT_REGISTRATION.out.reference
    } else {
        ANAT_REGISTRATION_PASSTHROUGH(anat_after_bias, config_file)
        anat_after_reg = ANAT_REGISTRATION_PASSTHROUGH.out.output
        anat_reg_transforms = ANAT_REGISTRATION_PASSTHROUGH.out.transforms
        anat_reg_reference = ANAT_REGISTRATION_PASSTHROUGH.out.reference
    }
    
    // ============================================
    // MASK TRANSFORMATION
    // ============================================
    // Transform brain mask to template space using registration transform
    // Input: anat_skull_mask: [sub, ses, brain_mask] (in conformed space)
    //        anat_reg_transforms: [sub, ses, bids_name, forward_xfm, inverse_xfm]
    //        anat_reg_reference: [sub, ses, reference_file]
    // Output: anat_skull_mask_registered: [sub, ses, transformed_mask] (in template space)
    // ============================================
    anat_skull_mask_registered = Channel.empty()
    if (registration_enabled && anat_skullstripping_enabled) {
        // Join mask with registration transform and reference
        def mask_transform_input = anat_skull_mask
            .join(anat_reg_transforms.map { sub, ses, bids_name, forward_xfm, inverse_xfm -> [sub, ses, forward_xfm] }, by: [0, 1])
            .join(anat_reg_reference, by: [0, 1])
            .map { sub, ses, mask_file, forward_xfm, reference ->
                [sub, ses, mask_file, forward_xfm, reference]
            }
        
        ANAT_APPLY_TRANSFORM_MASK(mask_transform_input, config_file)
        anat_skull_mask_registered = ANAT_APPLY_TRANSFORM_MASK.out.output
    }

    // ============================================
    // Backproject atlases to T1w space
    // ============================================
    // Only when registration is enabled (no backproject on passthrough)
    // Input: anat_reg_transforms [sub, ses, bids_name, forward, inverse]
    //        anat_after_bias [sub, ses, anat_file, bids_name]
    // Output: anat/atlas_space-T1w/*.nii.gz
    // ============================================
    // Custom templates carry no bundled atlases, so backprojection would only spawn no-op
    // tasks; skip it (is_custom_template resolved once above). The empty output channel
    // cascades safely: scanner input and surface projection become empty, and ARM6 reaches
    // surface reconstruction via join(..., remainder: true), which keeps every subject.
    anat_backproject_atlases_out = Channel.empty()
    if (registration_enabled && !is_custom_template) {
        def anat_after_bias_by_bids = anat_after_bias
            .map { sub, ses, anat_file, bids_name -> [sub, ses, bids_name, anat_file] }
        def backproject_input = anat_reg_transforms
            .join(anat_after_bias_by_bids, by: [0, 1, 2])
            .map { sub, ses, bids_name, forward_xfm, inverse_xfm, t1w_ref ->
                [sub, ses, bids_name, inverse_xfm, t1w_ref]
            }
        ANAT_BACKPROJECT_ATLASES_TO_T1W(backproject_input, config_file)
        anat_backproject_atlases_out = ANAT_BACKPROJECT_ATLASES_TO_T1W.out.output
    }

    // ============================================
    // Backproject atlases to scanner space
    // ============================================
    // Input: anat_backproject_atlases_out [sub, ses, atlas_files, bids_name]
    //        anat_conform_transforms [sub, ses, forward_xfm, inverse_xfm]
    //        anat_preconform [sub, ses, anat_file, bids_name]
    // Output: anat/atlas_space-scanner/*.nii.gz
    // ============================================
    anat_backproject_atlases_scanner_out = Channel.empty()
    if (registration_enabled && !is_custom_template) {
        def anat_preconform_by_bids = anat_preconform
            .map { sub, ses, anat_file, bids_name -> [sub, ses, bids_name, anat_file] }
        def t1w_atlases_input = anat_backproject_atlases_out
            .map { sub, ses, atlas_files, bids_name -> [sub, ses, bids_name, atlas_files] }
        def scanner_backproject_input = t1w_atlases_input
            .join(anat_preconform_by_bids, by: [0, 1, 2])
            .join(anat_conform_transforms, by: [0, 1])
            .map { sub, ses, bids_name, atlas_files, scanner_ref, forward_xfm, conform_inverse_xfm ->
                [sub, ses, bids_name, atlas_files, conform_inverse_xfm, scanner_ref]
            }
        ANAT_BACKPROJECT_ATLASES_TO_SCANNER(scanner_backproject_input, config_file)
        anat_backproject_atlases_scanner_out = ANAT_BACKPROJECT_ATLASES_TO_SCANNER.out.output
    }

    // ARM6 atlas extraction for surface reconstruction
    // Input: anat_backproject_atlases_out [sub, ses, atlas_files, bids_name]
    // Output: anat_arm6_atlas [sub, ses, arm6_atlas_file]
    def anat_arm6_atlas = Channel.empty()
    if (registration_enabled) {
        anat_arm6_atlas = anat_backproject_atlases_out
            .map { sub, ses, atlas_files, bids_name ->
                def atlas_list = atlas_files instanceof List ? atlas_files : (atlas_files ? [atlas_files] : [])
                def arm6_atlas = atlas_list.find { atlas_file -> atlas_file.getName().startsWith('atlas-ARM6') }
                [sub, ses, arm6_atlas]
            }
    }
    
    // ============================================
    // QUALITY CONTROL
    // ============================================
    // Generate QC reports for anatomical processing steps
    // ============================================
    if (anat_conform_enabled) {
        // ANAT_CONFORM emits full_fov as `optional: true`, so a subject whose full-FOV
        // helper raised produces nothing on that channel. A plain inner join would then
        // drop that subject from conform QC entirely, losing the figure that actually
        // matters. remainder: true keeps the row and pads it with nulls, which are
        // replaced by empty .dummy sentinels that QC_CONFORM's _real() maps back to None
        // -- Python then renders the standard figure alone. Kept as a remainder join on
        // the existing chain rather than a key-driven one so the pairwise semantics are
        // untouched when a session carries more than one conformed anatomical. No
        // right-only rows are possible: full_fov is emitted by the same process as
        // `output`, so its keys are always a subset.
        def dummy_full_fov = file("${workDir}/dummy_conform_full_fov.dummy").tap { it.toFile().text = "" }
        anat_after_conform
            .join(anat_conform_reference, by: [0, 1])
            .join(anat_conform_full_fov, by: [0, 1], remainder: true)
            .map { row ->
                [
                    row[0], row[1], row[2], row[3], row[4],
                    (row.size() > 5 && row[5]) ? row[5] : dummy_full_fov,
                    (row.size() > 6 && row[6]) ? row[6] : dummy_full_fov,
                    (row.size() > 7 && row[7]) ? row[7] : dummy_full_fov,
                ]
            }
            .set { conform_qc_input }
        QC_CONFORM(conform_qc_input, config_file)
    }
    
    if (anat_skullstripping_enabled) {
        anat_after_conform
            .join(anat_skull_mask, by: [0, 1])
            .map { sub, ses, anat_file, bids_name, mask_file ->
                [sub, ses, anat_file, mask_file, bids_name]
            }
            .set { skull_qc_input }
        QC_SKULLSTRIPPING(skull_qc_input, config_file)
        
        anat_after_skull_brain
            .join(anat_skull_seg, by: [0, 1])
            .map { sub, ses, brain_file, bids_name, seg_file ->
                [sub, ses, brain_file, seg_file, bids_name]
            }
            .set { atlas_qc_input }
        QC_ATLAS_SEGMENTATION(atlas_qc_input, config_file)
    }

    if (anat_bias_correction_enabled) {
        // For QC, compare T1w_brain before and after bias correction
        // Before: use skull-stripped brain from skullstripping (anat_after_skull_brain)
        // After: use bias-corrected T1w_brain (anat_after_bias_brain) - filter out dummy brains
        // Only generate QC when real bias-corrected brain is available
        // Both channels have structure [sub, ses, file, bids_name] (4 elements)
        def bias_qc_input = anat_after_skull_brain
            .join(anat_after_bias_brain.filter { sub, ses, brain_file, bids_name ->
                !brain_file.toString().contains('.dummy')
            }, by: [0, 1])
            .map { sub, ses, before_brain_file, before_bids, after_brain_file, after_bids ->
                [sub, ses, before_brain_file, after_brain_file, after_bids]
            }
        QC_BIAS_CORRECTION(bias_qc_input, config_file)
    }

    if (registration_enabled) {
        anat_after_reg
            .join(anat_reg_reference, by: [0, 1])
            .map { sub, ses, registered_file, bids_name, reference_file ->
                [sub, ses, registered_file, reference_file, bids_name]
            }
            .set { registration_qc_input }

        QC_REGISTRATION(registration_qc_input, config_file)
    }
    
    // ============================================
    // T2W PROCESSING
    // ============================================
    // All T2w already passed through ANAT_SYNTHESIS_T2W (see anat_t2w_all_jobs above).
    // Flow: T2w→T1w registration → apply conform → bias correction → apply registration
    //       OR: STOP after anatomical selection (for T2w without T1w)
    // ============================================
    // Step 1: Use T2w synthesis output directly
    // Input: anat_t2w_all_jobs: [sub, ses, anat_file, bids_name, needs_t1w_reg]
    // Output: t2w_preselection: [sub, ses, anat_file, bids_name, needs_t1w_reg]
    def t2w_preselection = anat_t2w_all_jobs
    
    // Step 3: Perform anatomical selection for ALL T2w files
    // Note: Bias correction is skipped for T2w - T2w proceeds directly from synthesis to anatomical selection
    // Use T1w before conform (anat_preconform) for T2w→T1w registration
    // This ensures both T2w and T1w are in their native space before registration
    // Note: anat_preconform is the full head version (_T1w), not the brain version (_T1w_brain)
    // Check for T1w matches for all T2w files
    // Priority: 1) Subject-level T1w (ses="", HIGHEST PRIORITY), 2) Same session T1w, 3) Cross-session T1w, 4) No T1w (stop processing)
    // Output: [sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses]
    def t2w_all_preselection = t2w_preselection
        .map { sub, ses, anat_file, bids_name, needs_t1w_reg ->
            [sub, ses, anat_file, bids_name]
        }
    
    def findUnmatchedT2w = channelHelpers.findUnmatchedT2w
    def t2w_anat_selection_raw = channelHelpers.performT2wAnatomicalSelection(
        t2w_all_preselection,
        anat_preconform,  // Full head T1w before conform, not brain (_T1w_brain)
        isT1wFile,
        findUnmatchedT2w
    )
    
    // Create copies for branching and QC (channel can only be consumed once)
    t2w_anat_selection_raw.multiMap { sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses ->
        for_branching: [sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses]
        for_qc: [sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses]
    }.set { t2w_anat_selection_channels }
    
    def t2w_anat_selection = t2w_anat_selection_channels.for_branching
    def t2w_anat_selection_for_qc = t2w_anat_selection_channels.for_qc
    
    // Step 4b: Split T2w into two paths based on T1w availability
    // Path 1: T2w with T1w (same-session or cross-session) - register to T1w space, then APPLY T1w's transforms
    // Path 2: T2w without T1w - stop after anatomical selection
    t2w_anat_selection
        .branch {
            with_t1w: it[4] != null  // t1w_file != null
            without_t1w: it[4] == null  // t1w_file == null
        }
        .set { t2w_branched_by_t1w }
    
    // Path 1: T2w with T1w - register to T1w space
    // Input: t2w_branched_by_t1w.with_t1w: [sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses]
    // Output: t2w_after_reg_to_t1w: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    def t2w_with_t1w_for_reg = t2w_branched_by_t1w.with_t1w
        .map { sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses ->
            [sub, ses, t2w_file, t2w_bids_name, t1w_file]
        }
        .multiMap { sub, ses, t2w_file, t2w_bids_name, t1w_file ->
            combined: [sub, ses, t2w_file, t2w_bids_name]
            reference: t1w_file
        }
        .set { t2w_reg_multi }
    
    // // debug print
    // t2w_reg_multi.combined.view() {
    //     println "|| t2w_reg_multi.combined ||: ${it}"
    // }
    // t2w_reg_multi.reference.view() {
    //     println "|| t2w_reg_multi.reference ||: ${it}"
    // }
    
    ANAT_T2W_TO_T1W_REGISTRATION(t2w_reg_multi.combined, t2w_reg_multi.reference, config_file)
    def t2w_after_reg_to_t1w = ANAT_T2W_TO_T1W_REGISTRATION.out.output
        .join(t2w_branched_by_t1w.with_t1w
            .map { sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses -> [sub, ses, anat_ses] }, 
            by: [0, 1])
        .map { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
            [sub, ses, t2w_file, t2w_bids_name, anat_ses]
        }
    
    // Path 2: T2w without T1w - stop after anatomical selection
    // Input: t2w_branched_by_t1w.without_t1w: [sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses]
    // Output: t2w_without_t1w_final: [sub, ses, t2w_file, t2w_bids_name]
    // (No further processing - stops here)
    def t2w_without_t1w_final = t2w_branched_by_t1w.without_t1w
        .map { sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses ->
            [sub, ses, t2w_file, t2w_bids_name]
        }
    
    // ============================================
    // Apply T1w's conform transform to T2w
    // ============================================
    // Input: t2w_after_reg_to_t1w: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    //        anat_conform_transforms: [sub, ses, forward_xfm, inverse_xfm] (keyed by T1w session)
    //        anat_conform_reference: [sub, ses, reference] (keyed by T1w session)
    // Output: t2w_after_apply_conform: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    def t2w_after_apply_conform = t2w_after_reg_to_t1w
    if (anat_conform_enabled) {
        // Join T2w with T1w's conform transform by anatomical session
        // all subject-level combinations, then filter keeps only matching anat_ses.
        def anat_conform_data = anat_conform_transforms
            .map { sub, ses, forward_xfm, inverse_xfm -> [sub, ses, forward_xfm] }
            .join(anat_conform_reference, by: [0, 1])
        
        def t2w_for_apply_conform = t2w_after_reg_to_t1w
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
                [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            }
            .combine(anat_conform_data, by: 0)
            .filter { sub, ses, t2w_file, t2w_bids_name, anat_ses, conform_ses, forward_xfm, reference ->
                matchSessions(anat_ses, conform_ses)
            }
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses, conform_ses, forward_xfm, reference ->
                [sub, ses, t2w_file, t2w_bids_name, forward_xfm, reference, anat_ses]
            }
        
        ANAT_APPLY_CONFORM(t2w_for_apply_conform, config_file)
        t2w_after_apply_conform = ANAT_APPLY_CONFORM.out.output
    } else {
        t2w_after_apply_conform = t2w_after_reg_to_t1w
    }
    
    // ============================================
    // T2W BIAS CORRECTION
    // ============================================
    // Apply bias correction to T2w using T1w's brain mask in conformed space
    // Input: t2w_after_apply_conform: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    //        anat_skull_mask: [sub, ses, mask_file] (keyed by T1w session)
    // Output: t2w_after_bias: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    //         t2w_after_bias_brain: [sub, ses, brain_file, bids_name, anat_ses]
    // ============================================
    // Use multiMap to create copies for bias correction and QC (channel can only be consumed once)
    t2w_after_apply_conform.multiMap { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
        for_bias: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
        for_qc: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    }.set { t2w_conform_channels }
    
    // Initialize - will be reassigned in if/else blocks
    def t2w_after_bias = Channel.empty()
    def t2w_before_bias_for_qc = t2w_conform_channels.for_qc
    def t2w_after_bias_brain = Channel.empty()
    def t2w_after_bias_for_qc_bias = Channel.empty()
    def t2w_after_bias_for_qc_snap1 = Channel.empty()
    
    if (anat_bias_correction_enabled) {
        // Join T2w with T1w's skull mask by anatomical session
        // Use combine() + filter() pattern (same as conform/registration)
        def t2w_with_mask = t2w_conform_channels.for_bias
            .combine(anat_skull_mask, by: 0)  // Combine by subject only
            .filter { sub, ses, t2w_file, t2w_bids_name, anat_ses, mask_ses, mask_file ->
                matchSessions(anat_ses, mask_ses)
            }
        
        // Branch: real mask vs dummy mask
        t2w_with_mask.branch {
            with_mask: !it[6].toString().contains('.dummy')
            no_mask: it[6].toString().contains('.dummy')
        }.set { t2w_bias_branched }
        
        // ----------------------------------------
        // Path 1: With real mask - run bias correction
        // ----------------------------------------
        // Use multiMap to create all needed channels from single consumption of branch
        t2w_bias_branched.with_mask.multiMap { sub, ses, t2w_file, t2w_bids_name, anat_ses, mask_ses, mask_file ->
            files: [sub, ses, t2w_file, t2w_bids_name]
            masks: [sub, ses, mask_file]
            anat_ses_lookup: [sub, ses, anat_ses, t2w_bids_name]  // Include bids_name for brain join
        }.set { t2w_with_mask_channels }
        
        ANAT_BIAS_CORRECTION_T2W(t2w_with_mask_channels.files, t2w_with_mask_channels.masks, config_file)
        
        // Join output with anat_ses lookup to restore anat_ses
        def t2w_with_mask_output = ANAT_BIAS_CORRECTION_T2W.out.output
            .join(t2w_with_mask_channels.anat_ses_lookup, by: [0, 1])
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses, orig_bids_name ->
                [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            }
        // Join brain with anat_ses and bids_name from lookup (use existing lookup channel, not output)
        def t2w_with_mask_brain = ANAT_BIAS_CORRECTION_T2W.out.brain
            .join(t2w_with_mask_channels.anat_ses_lookup, by: [0, 1])
            .map { sub, ses, brain_file, anat_ses, t2w_bids_name ->
                [sub, ses, brain_file, t2w_bids_name, anat_ses]
            }
        
        // ----------------------------------------
        // Path 2: No mask (dummy) - passthrough
        // ----------------------------------------
        // Use multiMap to create all needed channels from single consumption of branch
        t2w_bias_branched.no_mask.multiMap { sub, ses, t2w_file, t2w_bids_name, anat_ses, mask_ses, mask_file ->
            files: [sub, ses, t2w_file, t2w_bids_name]
            anat_ses_lookup: [sub, ses, anat_ses, t2w_bids_name]  // Include bids_name for brain join
        }.set { t2w_no_mask_channels }
        
        ANAT_BIAS_CORRECTION_PASSTHROUGH_T2W(t2w_no_mask_channels.files, config_file)
        
        // Join output with anat_ses lookup to restore anat_ses
        def t2w_no_mask_output = ANAT_BIAS_CORRECTION_PASSTHROUGH_T2W.out.output
            .join(t2w_no_mask_channels.anat_ses_lookup, by: [0, 1])
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses, orig_bids_name ->
                [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            }
        // Join brain with anat_ses and bids_name from lookup (use existing lookup channel, not output)
        def t2w_no_mask_brain = ANAT_BIAS_CORRECTION_PASSTHROUGH_T2W.out.brain
            .join(t2w_no_mask_channels.anat_ses_lookup, by: [0, 1])
            .map { sub, ses, brain_file, anat_ses, t2w_bids_name ->
                [sub, ses, brain_file, t2w_bids_name, anat_ses]
            }
        
        // ----------------------------------------
        // Mix outputs from both paths
        // ----------------------------------------
        def t2w_bias_mixed = t2w_with_mask_output.mix(t2w_no_mask_output)
        t2w_after_bias_brain = t2w_with_mask_brain.mix(t2w_no_mask_brain)
        
        // Create multiple copies for: publish, registration, QC bias, QC snap1
        // (channels can only be consumed once)
        t2w_bias_mixed.multiMap { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
            for_publish: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            for_registration: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            for_qc_bias: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            for_qc_snap1: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
        }.set { t2w_after_bias_channels }
        
        t2w_after_bias = t2w_after_bias_channels.for_registration
        t2w_after_bias_for_qc_bias = t2w_after_bias_channels.for_qc_bias
        t2w_after_bias_for_qc_snap1 = t2w_after_bias_channels.for_qc_snap1
        
        // ----------------------------------------
        // Publish T2w Phase 1 outputs (desc-preproc naming)
        // ----------------------------------------
        def t2w_publish_input = t2w_after_bias_channels.for_publish
            .join(t2w_after_bias_brain, by: [0, 1])
            .filter { sub, ses, t2w_file, t2w_bids, anat_ses, brain_file, brain_bids, brain_anat_ses ->
                // Only publish if not dummy
                !t2w_file.toString().contains('.dummy')
            }
            .map { sub, ses, t2w_file, t2w_bids, anat_ses, brain_file, brain_bids, brain_anat_ses ->
                [sub, ses, t2w_file, brain_file, t2w_bids]
            }
        
        ANAT_PUBLISH_T2W(t2w_publish_input, config_file)
    } else {
        // Bias correction disabled - pass through
        // Create copies for registration and QC
        t2w_conform_channels.for_bias.multiMap { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
            for_registration: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            for_qc_snap1: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
        }.set { t2w_passthrough_channels }
        
        t2w_after_bias = t2w_passthrough_channels.for_registration
        t2w_after_bias_for_qc_snap1 = t2w_passthrough_channels.for_qc_snap1
    }

    // ============================================
    // Apply T1w's registration transform to T2w
    // ============================================
    // Input: t2w_after_bias: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    //        anat_reg_transforms: [sub, ses, bids_name, forward_xfm, inverse_xfm] (keyed by T1w session)
    //        anat_reg_reference: [sub, ses, reference] (keyed by T1w session)
    // Output: t2w_after_apply_reg: [sub, ses, t2w_file, t2w_bids_name]
    // ============================================
    def t2w_after_apply_reg = t2w_after_bias
    if (registration_enabled) {
        def anat_reg_data = anat_reg_transforms
            .map { sub, ses, bids_name, forward_xfm, inverse_xfm -> [sub, ses, forward_xfm] }
            .join(anat_reg_reference, by: [0, 1])
        
        def t2w_for_apply_reg = t2w_after_bias
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
                [sub, ses, t2w_file, t2w_bids_name, anat_ses]
            }
            .combine(anat_reg_data, by: 0)  // Combine by subject only
            .filter { sub, ses, t2w_file, t2w_bids_name, anat_ses, reg_ses, forward_xfm, reference ->
                matchSessions(anat_ses, reg_ses)
            }
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses, reg_ses, forward_xfm, reference ->
                [sub, ses, t2w_file, t2w_bids_name, forward_xfm, reference]
            }
        
        ANAT_APPLY_TRANSFORMATION(t2w_for_apply_reg, config_file)
        t2w_after_apply_reg = ANAT_APPLY_TRANSFORMATION.out.output
    } else {
        t2w_after_apply_reg = t2w_after_bias
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
                [sub, ses, t2w_file, t2w_bids_name]
            }
    }
    
    // ============================================
    // Generate T1wT2wCombined image (only channels that have the same t2w_ses as t1w_ses)
    // ============================================
    // Generate T1wT2wCombined image using T1w, T2w, and segmentation
    // This is an enhanced T1w image, so output is keyed by T1w session (not T2w session)
    // Input channels:
    //   - t2w_after_bias: [sub, t2w_ses, t2w_file, t2w_bids_name, anat_ses]
    //   - ANAT_PUBLISH_PHASE1.out.output (filtered): [sub, t1w_ses, t1w_file, t1w_bids_name]
    //   - anat_skull_seg: [sub, t1w_ses, seg_file]
    //   - anat_skull_seg_lut: [sub, t1w_ses, lut_file]
    // Output: [sub, t1w_ses, combined_file]
    // Only runs when: skullstripping enabled (segmentation available) AND both T1w and T2w available
    // ============================================
    // Initialize unified channel (will be populated if skullstripping is enabled)
    // Structure: [sub, ses, file, bids_name] - T1wT2wCombined if available, otherwise T1w
    def anat_after_t1wt2wcombined = Channel.empty()
    
    if (anat_skullstripping_enabled) {
        // Get T1w files from published Phase 1 outputs (used for combined processing and QC)
        // Input: [sub, t1w_ses, t1w_file, t1w_bids_name]
        def t1w_after_bias = ANAT_PUBLISH_PHASE1.out.output
            .filter(isT1wFile)
        
        // ============================================
        // Build input for T1wT2wCombined process
        // ============================================
        // Input: [sub, t2w_ses, t2w_file, t2w_bids_name, anat_ses]
        def t1wt2w_combined_input = t2w_after_bias
            // Combine with T1w files by subject only (multiple T2w sessions may reference same T1w)
            .combine(t1w_after_bias, by: 0)
            // Filter to keep only where T2w's anat_ses matches T1w session
            .filter { sub, t2w_ses, t2w_file, t2w_bids_name, anat_ses, t1w_ses, t1w_file, t1w_bids_name ->
                // Require that:
                // 1) The chosen anatomical session for this T2w (anat_ses) matches the T1w session, AND
                // 2) The T2w's own session matches the chosen anatomical session
                matchSessions(anat_ses, t1w_ses) && matchSessions(t2w_ses, anat_ses)
            }
            // Reorganize and key by T1w session (critical!)
            // Input: [sub, t2w_ses, t2w_file, t2w_bids_name, anat_ses, t1w_ses, t1w_file, t1w_bids_name]
            // Output: [sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file]
            .map { sub, t2w_ses, t2w_file, t2w_bids_name, anat_ses, t1w_ses, t1w_file, t1w_bids_name ->
                [sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file]
            }
            // Combine with segmentation by subject only
            .combine(anat_skull_seg, by: 0)
            .filter { sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_ses, seg_file ->
                matchSessions(t1w_ses, seg_ses)
            }
            .map { sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_ses, seg_file ->
                [sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_file]
            }
            // Combine with LUT by subject only
            .combine(anat_skull_seg_lut, by: 0)
            .filter { sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_file, lut_ses, lut_file ->
                matchSessions(t1w_ses, lut_ses)
            }
            .map { sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_file, lut_ses, lut_file ->
                [sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_file, lut_file]
            }

        // Process receives: [sub, t1w_ses, t1w_file, t1w_bids_name, t2w_file, seg_file, lut_file]
        // Process outputs: [sub, t1w_ses, combined_file, t1w_bids_name]
        ANAT_T1WT2W_COMBINED(t1wt2w_combined_input, config_file)
        
        // ============================================
        // Create unified channel with passthrough
        // ============================================
        // Structure: [sub, ses, file, bids_name]
        def t1wt2w_combined_output = ANAT_T1WT2W_COMBINED.out.output
        
        // Create passthrough channel for T1w sessions without T2w
        // Strategy: Use left anti-join pattern with mix() + groupTuple()
        // Extract [sub, ses] pairs from combined output
        def t1wt2w_sessions = t1wt2w_combined_output
            .map { sub, ses, file, bids_name -> [sub, ses] }
            .unique()
        
        // Tag all T1w sessions and matched sessions, then find unmatched
        def t1w_keys_tagged = t1w_after_bias
            .map { sub, ses, t1w_file, t1w_bids_name -> [[sub, ses], 't1w'] }
        
        def matched_keys_tagged = t1wt2w_sessions
            .map { sub, ses -> [[sub, ses], 'matched'] }
        
        // Find unmatched T1w sessions (those without combined output)
        def unmatched_keys = t1w_keys_tagged
            .mix(matched_keys_tagged)
            .groupTuple(by: 0)
            .filter { key, tags -> !tags.contains('matched') }
            .map { key, tags -> key }  // key is [sub, ses]
        
        // Join unmatched keys back with t1w_after_bias to get full data
        def t1w_passthrough = unmatched_keys
            .join(t1w_after_bias, by: [0, 1])
        
        // Mix combined and passthrough to create unified channel
        anat_after_t1wt2wcombined = t1wt2w_combined_output.mix(t1w_passthrough)
        
        // ============================================
        // QC: T1wT2wCombined comparison
        // ============================================
        // Generate QC snapshot comparing T1w after bias correction vs T1wT2wCombined
        // Input: ANAT_T1WT2W_COMBINED.out.output: [sub, t1w_ses, combined_file, t1w_bids_name]
        //        t1w_after_bias: [sub, t1w_ses, t1w_file, t1w_bids_name]
        //        anat_skull_mask: [sub, t1w_ses, mask_file]
        // ============================================
        def t1wt2w_qc_input = ANAT_T1WT2W_COMBINED.out.output
            .join(t1w_after_bias, by: [0, 1])  // Join by subject and T1w session
            .map { sub, t1w_ses, combined_file, combined_bids_name, t1w_file, t1w_bids_name ->
                // Use original T1w bids_name for output naming
                [sub, t1w_ses, t1w_file, combined_file, t1w_bids_name]
            }
            .join(anat_skull_mask, by: [0, 1])  // Join with mask by subject and T1w session
            .map { sub, t1w_ses, t1w_file, combined_file, t1w_bids_name, mask_file ->
                [sub, t1w_ses, t1w_file, combined_file, mask_file, t1w_bids_name]
            }
        
        QC_T1WT2W_COMBINED(t1wt2w_qc_input, config_file)
    }

    // ============================================
    // T2W QUALITY CONTROL
    // ============================================
    // Three QC snaps for T2w with T1w:
    // 1. T2w bias correction QC: Before/after bias correction comparison
    // 2. T2w→T1w registration QC: T2w in T1w space (after bias correction) with T1w brain contours
    // 3. T2w in template space QC: T2w in template space (after all transforms) with template contours
    // ============================================
    
    // Filter T1w brain images for QC
    def t1w_brain = anat_after_skull_brain
        .filter(isT1wFile)
    
    // ============================================
    // QC SNAP 0: T2w Bias Correction
    // ============================================
    // Compare T2w before and after bias correction
    // Input: t2w_before_bias_for_qc: [sub, ses, t2w_file, t2w_bids_name, anat_ses] (before)
    //        t2w_after_bias: [sub, ses, t2w_file, t2w_bids_name, anat_ses] (after)
    // ============================================
    if (anat_bias_correction_enabled) {
        def t2w_bias_qc_input = t2w_before_bias_for_qc
            .map { sub, ses, t2w_file, t2w_bids_name, anat_ses -> [sub, ses, t2w_file, t2w_bids_name] }
            .join(t2w_after_bias_for_qc_bias
                .filter { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
                    // Only generate QC for real bias-corrected files (not passthrough)
                    !t2w_file.toString().contains('.dummy') && t2w_file.toString().contains('desc-biascorrect')
                }
                .map { sub, ses, t2w_file, t2w_bids_name, anat_ses -> [sub, ses, t2w_file, t2w_bids_name] },
                by: [0, 1])
            .map { sub, ses, before_file, before_bids, after_file, after_bids ->
                [sub, ses, before_file, after_file, after_bids]
            }
        
        QC_BIAS_CORRECTION_T2W(t2w_bias_qc_input, config_file)
    }
    
    // ============================================
    // QC SNAP 1: T2w→T1w Registration
    // ============================================
    // Underlay: Bias-corrected T2w (after bias correction)
    // Overlay: T1w brain - contours will be generated
    // Input: t2w_after_bias_for_qc_snap1: [sub, ses, t2w_file, t2w_bids_name, anat_ses]
    //        t1w_brain: [sub, ses, t1w_file, t1w_bids_name] (keyed by anatomical session)
    // ============================================
    // IMPORTANT: Use combine() + filter() instead of join() because multiple T2w sessions
    // may reference the SAME T1w session. join() causes race condition.
    def t2w_qc1_input = t2w_after_bias_for_qc_snap1
        .map { sub, ses, t2w_file, t2w_bids_name, anat_ses ->
            [sub, ses, t2w_file, t2w_bids_name, anat_ses]
        }
        .combine(t1w_brain, by: 0)  // Combine by subject only
        .filter { sub, ses, t2w_file, t2w_bids_name, anat_ses, t1w_ses, t1w_file, t1w_bids_name ->
            matchSessions(anat_ses, t1w_ses)  // Keep only where T2w's anat_ses matches T1w session
        }
        .map { sub, ses, t2w_file, t2w_bids_name, anat_ses, t1w_ses, t1w_file, t1w_bids_name ->
            [sub, ses, t2w_file, t1w_file, t2w_bids_name]
        }
    
    t2w_qc1_input
        .multiMap { sub, ses, t2w_file, t1w_file, t2w_bids_name ->
            combined: [sub, ses, t2w_file, t1w_file]
            bids_name: t2w_bids_name
        }
        .set { t2w_qc1_channels }
    
    // Call QC process - it will handle empty channels gracefully
    QC_T2W_TO_T1W_REGISTRATION(t2w_qc1_channels.combined, t2w_qc1_channels.bids_name, config_file)
    
    // ============================================
    // QC SNAP 2: T2w in Template Space
    // ============================================
    // Underlay: T2w after all transforms (in template space)
    // Overlay: Template reference - contours will be generated
    // Input: t2w_after_apply_reg: [sub, ses, t2w_file, t2w_bids_name]
    //        t2w_anat_selection_for_qc: [sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses] (to get anat_ses)
    //        anat_reg_reference: [sub, ses, reference] (keyed by anatomical session)
    // ============================================
    def t2w_qc2_input = t2w_after_apply_reg
        .join(t2w_anat_selection_for_qc
            .filter { sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses -> t1w_file != null }
            .map { sub, ses, t2w_file, t2w_bids_name, t1w_file, anat_ses -> [sub, ses, anat_ses] },
            by: [0, 1])
        .map { sub, ses, registered_t2w_file, t2w_bids_name, anat_ses ->
            [sub, ses, registered_t2w_file, t2w_bids_name, anat_ses]
        }
        .combine(anat_reg_reference, by: 0)  // Combine by subject only
        .filter { sub, ses, registered_t2w_file, t2w_bids_name, anat_ses, ref_ses, template_ref ->
            matchSessions(anat_ses, ref_ses)  // Keep only where T2w's anat_ses matches reference session
        }
        .map { sub, ses, registered_t2w_file, t2w_bids_name, anat_ses, ref_ses, template_ref ->
            [sub, ses, registered_t2w_file, template_ref, t2w_bids_name]
        }
    
    t2w_qc2_input
        .multiMap { sub, ses, t2w_file, template_ref, t2w_bids_name ->
            combined: [sub, ses, t2w_file, template_ref]
            bids_name: t2w_bids_name
        }
        .set { t2w_qc2_channels }
    
    // Call QC process - it will handle empty channels gracefully
    QC_T2W_TEMPLATE_SPACE(t2w_qc2_channels.combined, t2w_qc2_channels.bids_name, config_file)
    
    // ============================================
    // SURFACE RECONSTRUCTION
    // ============================================
    def anat_for_surf_recon = use_t1wt2wcombined ? anat_after_t1wt2wcombined : anat_after_bias
    SURF_RECON_WF(anat_for_surf_recon, anat_skull_seg, anat_skull_mask, anat_arm6_atlas, gpu_queue)

    def surf_recon_enabled_for_emit = paramResolver.getYamlBool("anat.surface_reconstruction.enabled")
    
    // No `def`: must be workflow-scoped so emit
    surf_actual_subject_id = (surf_recon_enabled_for_emit && anat_skullstripping_enabled)
        ? SURF_RECON_WF.out.surf_actual_subject_id_ch
        : Channel.empty()

    // ============================================
    // Project atlases onto FastSurfer surfaces
    // ============================================
    // Only when registration produced T1w atlases AND surf recon ran successfully.
    // Input: anat_backproject_atlases_out [sub, ses, atlas_files, bids_name]
    //        surf_subject_dir            [sub, ses, fastsurfer_subject_dir]
    // Output: anat/atlas_space-fsnative/{*.nii.gz, *_hemi-{L,R}_*.func.gii}
    // Joining on the FastSurfer subject_dir (a real task-output path) both serializes
    // this after surf recon AND stages the tree into the work dir, so we never race the
    // async publishDir copy. Subjects whose surf recon failed emit no dir and are dropped.
    // ============================================
    def surf_subject_dir = (surf_recon_enabled_for_emit && anat_skullstripping_enabled)
        ? SURF_RECON_WF.out.surf_subject_dir_ch
        : Channel.empty()
    // Skip for custom templates: no bundled atlases means zero atlas files to project, and
    // the process would otherwise still publish only a status JSON, creating an empty
    // atlas_space-fsnative/ directory.
    if (registration_enabled && surf_recon_enabled_for_emit && anat_skullstripping_enabled && !is_custom_template) {
        // join yields [sub, ses, bids_name, atlas_files, fs_subject_dir] = the process input tuple.
        def surf_atlas_input = anat_backproject_atlases_out
            .map { sub, ses, atlas_files, bids_name -> [sub, ses, bids_name, atlas_files] }
            .join(surf_subject_dir, by: [0, 1])
        ANAT_PROJECT_ATLASES_TO_SURFACE(surf_atlas_input, config_file)
    }

    // ============================================
    // COLLECT QC CHANNELS
    // ============================================
    anat_qc_channels = Channel.empty()
    if (registration_enabled) {
        anat_qc_channels = anat_qc_channels.mix(QC_REGISTRATION.out.metadata)
    }
    if (anat_conform_enabled) {
        anat_qc_channels = anat_qc_channels.mix(QC_CONFORM.out.metadata)
    }
    if (anat_bias_correction_enabled) {
        anat_qc_channels = anat_qc_channels.mix(QC_BIAS_CORRECTION.out.metadata)
        anat_qc_channels = anat_qc_channels.mix(QC_BIAS_CORRECTION_T2W.out.metadata)
    }
    if (anat_skullstripping_enabled) {
        anat_qc_channels = anat_qc_channels.mix(QC_SKULLSTRIPPING.out.metadata)
        anat_qc_channels = anat_qc_channels.mix(QC_ATLAS_SEGMENTATION.out.metadata)
    }
    anat_qc_channels = anat_qc_channels.mix(QC_T2W_TO_T1W_REGISTRATION.out.metadata)
    anat_qc_channels = anat_qc_channels.mix(QC_T2W_TEMPLATE_SPACE.out.metadata)
    anat_qc_channels = anat_qc_channels.mix(SURF_RECON_WF.out.surf_qc_channels)
    
    // ============================================
    // EMIT OUTPUT CHANNELS
    // ============================================
    emit:
    anat_after_bias_brain
    anat_reg_transforms
    anat_reg_reference
    surf_actual_subject_id
    anat_subjects_ch
    anat_qc_channels
    anat_skull_seg
    anat_skull_seg_lut
}
