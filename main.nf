/*
 * Main Nextflow workflow for brainana
 * 
 * This workflow processes BIDS datasets using per-step parallelization
 * for maximum efficiency.
 * 
 * BIDS discovery is performed by a Python script BEFORE this workflow runs.
 * The discovery script validates the BIDS dataset, discovers all jobs, and
 * saves JSON files that this workflow reads to create channels.
 * 
 */

nextflow.enable.dsl=2

import groovyx.gpars.dataflow.DataflowQueue

// Include sub-workflows
include { ANAT_WF } from './workflows/anatomical_workflow.nf'
include { FUNC_WF } from './workflows/functional_workflow.nf'

// QC reports are generated in workflow.onComplete (below) so that a report is
// always produced, even when the run aborts early on an upstream failure.

// Load parameter resolver
def paramResolver = evaluate(new File("${projectDir}/workflows/param_resolver.groovy").text)

workflow {
    // ============================================
    // INITIALIZE PARAMETER RESOLVER
    // ============================================
    // Initialize parameter resolver with priority: CLI params → YAML config → defaults.yaml
    paramResolver.initialize(params, projectDir)

    // ============================================
    // VALIDATE PARAMETER NAMES (fail fast on unknown flags)
    // ============================================
    // Nextflow silently turns any --foo on the command line into params.foo, so a
    // mistyped flag name would otherwise be ignored and the run would proceed with
    // defaults (e.g. --custom-template silently doing nothing while the default
    // template is used). This is the BACKSTOP guard (the bash entry layers reject
    // unknown flags earlier); it also covers direct `nextflow run main.nf` invocations.
    //
    // The accepted names come from the shared allowlist known_flags.txt (single source of
    // truth, also read by flags.sh). Fail-open: if the file is missing, skip name
    // validation rather than risk aborting an otherwise-valid run.
    def knownFlagsFile = new File("${projectDir}/known_flags.txt")
    def KNOWN_PARAMS = knownFlagsFile.isFile()
        ? (knownFlagsFile.readLines()
             .collect { it.replaceAll(/#.*/, '').trim() }
             .findAll { it } as Set)
        : ([] as Set)
    def unknownParams = KNOWN_PARAMS.isEmpty()
        ? []
        : params.keySet().findAll { !KNOWN_PARAMS.contains(it as String) }
    if (unknownParams) {
        def flags = unknownParams.collect { "--${it}" }.sort().join(', ')
        // Reuse the canonical usage listing (same text as --help) so the error is a full,
        // standard argument reference. Falls back to a short synopsis if USAGE.txt is absent.
        def usageFile = new File("${projectDir}/USAGE.txt")
        def usage = usageFile.isFile()
            ? usageFile.text.trim()
            : "usage: docker run ... <image> [bids_dir] [output_dir] [OPTIONS]\n" +
              "run with --help for the full argument list."
        error "Unknown argument(s): ${flags}\n\n" +
              "${usage}\n\n" +
              "Hint: custom templates use --output_space <file>, not --custom-template. " +
              "The run_brainana.sh wrapper normalizes hyphenated spellings (e.g. --output-space) " +
              "to underscore; when invoking nextflow directly, use the underscore form (--output_space)."
    }

    // ============================================
    // VALIDATE CUSTOM TEMPLATE (fail fast, no silent fallback)
    // ============================================
    // When output_space is a custom template file path, reject it up front if the
    // extension is not .nii/.nii.gz or the file does not exist, so the run aborts
    // immediately with a clear message rather than failing deep inside a process.
    // Mirrors is_custom_template_path()/resolve_template() in utils/templates.py.
    def _output_space = paramResolver.getParamOutputSpace(params, 'output_space')
    if (_output_space) {
        def _os = _output_space.toString().trim()
        def _looks_like_path = _os.contains('/') || _os.endsWith('.nii') || _os.endsWith('.nii.gz')
        if (_looks_like_path) {
            if (!(_os.endsWith('.nii') || _os.endsWith('.nii.gz'))) {
                error "Custom template must be a .nii or .nii.gz file, got: '${_os}'"
            }
            if (!new File(_os).isFile()) {
                error "Custom template file not found: '${_os}'"
            }
        }
    }

    // ============================================
    // GENERATE EFFECTIVE CONFIG FILE
    // ============================================
    // Generate effective config.yaml that merges: CLI params → YAML config → defaults.yaml
    // This file will be used by all processes instead of passing individual parameters
    // Must be generated before workflows are invoked
    def effective_config_path = paramResolver.generateEffectiveConfig(params, projectDir, params.output_dir)
    
    // Verify the file was created successfully
    def effective_config_file_check = new File(effective_config_path)
    if (!effective_config_file_check.exists()) {
        error "Failed to generate effective config file at: ${effective_config_path}"
    }
    
    def effective_config_file = file(effective_config_path)

    // ============================================
    // WRITE dataset_description.json (BIDS derivatives root)
    // ============================================
    // Records brainana + version and the run's template source (bundled spec or
    // custom template file path) once for the whole dataset. Best-effort: never
    // aborts the run (the script always exits 0).
    if (params.output_dir) {
        def ddScript = "${projectDir}/src/nhp_mri_prep/nextflow_scripts/write_dataset_description.py"
        try {
            def ddProc = [params.python_exe ?: 'python3', ddScript,
                          '--output-dir', params.output_dir.toString(),
                          '--config-file', effective_config_path.toString()].execute()
            ddProc.waitFor()
            if (ddProc.exitValue() != 0) {
                log.warn "dataset_description.json generation exited ${ddProc.exitValue()}: ${ddProc.err.text}"
            }
        } catch (Exception e) {
            log.warn "Could not write dataset_description.json: ${e.message}"
        }
    }

    // ============================================
    // RESOLVE PARAMETERS (for workflow logic)
    // ============================================
    // Get anat_only parameter with priority: CLI → YAML → defaults.yaml
    def anat_only = paramResolver.getParamBool(params, 'anat_only')
    
    // Resolve GPU usage policy from config + runtime hardware detection.
    // general.gpu_device controls whether GPU scheduling should be enabled.
    // -1 / "cpu" => force CPU-only scheduling (no GPU tokens consumed by workflows).
    // Workflows call paramResolver.resolveUseGpu(params) themselves; do not assign
    // params.use_gpu here (a runtime assignment would not override nextflow.config).
    def use_gpu = paramResolver.resolveUseGpu(params)
    log.info "GPU scheduling: ${use_gpu ? 'enabled' : 'disabled'} (${params.gpu_count ?: 0} GPU(s) detected)"

    // ============================================
    // GLOBAL GPU TOKEN POOL
    // ============================================
    // Create a shared GPU token queue so ALL GPU processes draw from the same pool
    // This enforces max_jobs_per_gpu across anatomical + functional GPU steps.
    def gpu_queue = new DataflowQueue()
    def gpu_count = use_gpu ? (params.gpu_count ?: 0) : 0
    def max_jobs_per_gpu = params.max_jobs_per_gpu ?: 1
    def token_gpu_count = gpu_count > 0 ? gpu_count : 1
    def token_jobs_per_gpu = max_jobs_per_gpu > 0 ? max_jobs_per_gpu : 1
    (0..<token_gpu_count).each { gpu_id ->
        (0..<token_jobs_per_gpu).each { gpu_queue << gpu_id }
    }
    
    // ============================================
    // RUN ANATOMICAL WORKFLOW
    // ============================================
    ANAT_WF(gpu_queue)

    // ============================================
    // RUN FUNCTIONAL WORKFLOW (conditionally)
    // ============================================
    if (!anat_only) {
        FUNC_WF(
            ANAT_WF.out.anat_after_bias_brain,  // Use brain version for functional registration
            ANAT_WF.out.anat_reg_transforms,
            ANAT_WF.out.anat_reg_reference,
            ANAT_WF.out.surf_actual_subject_id,
            ANAT_WF.out.anat_skull_seg,       // T1w-space segmentation (for confounds tissue regressors)
            ANAT_WF.out.anat_skull_seg_lut,   // atlas LUT (tissue classification)
            gpu_queue
        )
    }
    
    // ============================================
    // QC REPORT GENERATION
    // ============================================
    // Reports are generated in workflow.onComplete (below) instead of as a DAG
    // node, so they are produced on both success and early abort. See main.nf
    // bottom for the handler.
}

// ============================================
// WORKFLOW COMPLETION HANDLER
// ============================================
// Always runs (success OR early abort). Generates the per-subject QC reports so
// the user gets a report even when the run aborts, and prints a failure summary.
workflow.onComplete {
    def failedCount = workflow.stats.failedCount
    def ignoredCount = workflow.stats.ignoredCount

    // --- Always (re)generate QC reports, embedding the run status -------------
    // QC snapshots are published incrementally, so a report built here contains
    // whatever completed. Wrapped so report generation can never crash the run.
    if (params.output_dir != null) {
        try {
            def traceFile = "${params.output_dir}/nextflow_reports/nextflow_trace.txt"
            def status = [
                success        : workflow.success,
                exit_status    : workflow.exitStatus,
                error_message  : workflow.errorMessage,
                error_report   : workflow.errorReport,
                duration       : workflow.duration?.toString(),
                succeeded_count: workflow.stats.succeededCount,
                failed_count   : failedCount,
                ignored_count  : ignoredCount,
                trace_file     : traceFile,
            ]

            def reportsDir = new File("${params.output_dir}/nextflow_reports")
            reportsDir.mkdirs()
            def statusFile = new File(reportsDir, "run_status.json")
            statusFile.text = groovy.json.JsonOutput.toJson(status)

            // Prefer the effective config (merged CLI + YAML + defaults) so the report
            // reflects actual run params — e.g. a CLI --output_space custom template path.
            // Fall back to the raw config/defaults if the effective config is absent.
            def effective_cfg = "${params.output_dir}/nextflow_reports/config.yaml"
            def config_file_path = new File(effective_cfg).exists() ? effective_cfg : (params.config_file ?: "${projectDir}/src/nhp_mri_prep/config/defaults.yaml")
            def python = System.getenv('PYTHON') ?: 'python3'
            def gen_script = "${projectDir}/src/nhp_mri_prep/nextflow_scripts/generate_reports.py"
            def cmd = [
                python, gen_script,
                "--output-dir", "${params.output_dir}",
                "--config-file", config_file_path,
                "--status-file", statusFile.toString(),
            ]
            def out = new StringBuffer()
            def err = new StringBuffer()
            def proc = cmd.execute()
            proc.consumeProcessOutput(out, err)
            proc.waitFor()
            if (out.length() > 0) print out.toString()
            if (proc.exitValue() != 0) {
                println "WARNING: QC report generation exited with ${proc.exitValue()}."
                if (err.length() > 0) println err.toString()
            }
        } catch (Exception e) {
            println "WARNING: Could not generate QC report - ${e.message}"
            println "  Check the trace file: ${params.output_dir}/nextflow_reports/nextflow_trace.txt"
        }
    }

    // --- Failure summary ------------------------------------------------------
    // Report on failed/ignored tasks (typically surface reconstruction failures)
    if (ignoredCount > 0 || failedCount > 0) {
        println ""
        println "WARNING: Some tasks failed (${failedCount + ignoredCount} total)."
        println "This may include surface reconstruction jobs that failed due to image quality issues."
        println ""
        println "To see details, check the trace file:"
        println "  ${params.output_dir}/nextflow_reports/nextflow_trace.txt"
        println ""
        println "Filter for failed tasks with:"
        println "  grep -E 'FAILED|ABORTED' ${params.output_dir}/nextflow_reports/nextflow_trace.txt"
    }

    if (!workflow.success) {
        println ""
        println "Pipeline aborted early. A partial QC report (with the error in the"
        println "Run status section) was written to: ${params.output_dir}/sub-*.html"
    }

    println ""
}
