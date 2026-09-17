/*
 * Surface Reconstruction Workflow
 *
 * Generates cortical surfaces and measurements from anatomical data.
 * Requires skullstripping and optionally uses T1wT2w combined images.
 *
 * Inputs from ANAT_WF: anat_for_surf_recon, anat_skull_seg, anat_skull_mask, anat_arm6_atlas
 */

nextflow.enable.dsl=2

// Include surface reconstruction modules
include { ANAT_SURFACE_RECONSTRUCTION } from '../modules/anatomical.nf'
include { QC_SURF_RECON_TISSUE_SEG } from '../modules/qc.nf'
include { QC_CORTICAL_SURF_AND_MEASURES } from '../modules/qc.nf'
include { ANAT_SURFACE_BASE_TEMPLATE } from '../modules/anatomical.nf'
include { ANAT_SURFACE_BASE_ATLAS } from '../modules/anatomical.nf'
include { ANAT_SURFACE_BASE_RECON } from '../modules/anatomical.nf'
include { ANAT_SURFACE_RECONSTRUCTION_LONG } from '../modules/anatomical.nf'
include { ANAT_SURFACE_LONG_CHANGE_STATS } from '../modules/anatomical.nf'
// Aliased: the base needs the same fsnative atlas projection the cross-sectional
// sessions get. Only the base -- every timepoint shares its mesh, so the base's
// vertex labels are theirs too, and a per-timepoint projection would collide with
// the cross-sectional filenames (get_bids_prefix drops space-, so the frame that
// distinguishes them is not in the stem it builds).
include { ANAT_PROJECT_ATLASES_TO_SURFACE as ANAT_PROJECT_ATLASES_TO_SURFACE_BASE } from '../modules/anatomical.nf'
// Aliased: a process can be invoked only once per workflow, and the base and
// longitudinal directories need the same QC as the cross-sectional ones.
include { QC_SURF_RECON_TISSUE_SEG as QC_SURF_RECON_TISSUE_SEG_LONG } from '../modules/qc.nf'
include { QC_CORTICAL_SURF_AND_MEASURES as QC_CORTICAL_SURF_AND_MEASURES_LONG } from '../modules/qc.nf'

// Load parameter resolver and config helpers
def paramResolver = evaluate(new File("${projectDir}/workflows/param_resolver.groovy").text)
def configHelpers = evaluate(new File("${projectDir}/workflows/config_helpers.groovy").text)

workflow SURF_RECON_WF {
    take:
    anat_for_surf_recon    // [sub, ses, anat_file, bids_name]
    anat_skull_seg         // [sub, ses, seg_file]
    anat_skull_mask        // [sub, ses, mask_file]
    anat_arm6_atlas        // [sub, ses, arm6_atlas_file]
    gpu_queue

    main:
    // ============================================
    // INITIALIZATION
    // ============================================
    configHelpers.ensureParamResolverInitialized(paramResolver, params, projectDir)
    def config_file_path = configHelpers.getEffectiveConfigPath(params, projectDir)
    def config_file = config_file_path

    // ============================================
    // RESOLVE PARAMETERS
    // ============================================
    def surf_recon_enabled = paramResolver.getYamlBool("anat.surface_reconstruction.enabled")
    def anat_skullstripping_enabled = paramResolver.getYamlBool("anat.skullstripping_segmentation.enabled")
    // Driven by synthesis_level alone -- deliberately no second config key, so
    // there is no way to reach an "enabled but nothing happened" state.
    def synthesis_level = paramResolver.getYamlString("anat.synthesis_level", "subject")
    def longitudinal_enabled = ("${synthesis_level}" == "session_longitudinal")

    // ============================================
    // SURFACE RECONSTRUCTION
    // ============================================
    surf_qc_channels = Channel.empty()
    // [sub, ses, fastsurfer_dir_name] for downstream func tSNR / QC (empty if surf recon skipped)
    // No `def`: must be workflow-scoped so emit: surf_actual_subject_id_ch resolves (Nextflow 25+).
    surf_actual_subject_id_ch = Channel.empty()
    // [sub, ses, fastsurfer_subject_dir] task-output path, staged (not read from output_dir)
    // by consumers so they don't race the async publishDir copy. Empty if surf recon skipped.
    surf_subject_dir_ch = Channel.empty()
    // Longitudinal stream. Empty unless synthesis_level is session_longitudinal.
    // These stay separate from surf_subject_dir_ch on purpose: functional
    // consumers must keep seeing the cross-sectional trees. A _long tree's
    // orig.mgz lives in base space, and project_tsnr_to_surface uses
    // `mri_vol2surf --regheader`, which assumes header agreement with the
    // session's own volumes -- so projecting session data onto longitudinal
    // surfaces would be misregistered by exactly the timepoint-to-base
    // transform.
    surf_base_dir_ch = Channel.empty()
    surf_base_subject_id_ch = Channel.empty()
    surf_long_subject_dir_ch = Channel.empty()
    surf_long_subject_id_ch = Channel.empty()

    if (surf_recon_enabled && anat_skullstripping_enabled) {
        // Step 0: Calculate session count per subject (for surface reconstruction naming)
        def anat_sessions_per_subject = anat_for_surf_recon
            .map { sub, ses, anat_file, bids_name ->
                [sub, ses]
            }
            .unique()
            .groupTuple(by: 0)
            .map { sub, ses_list ->
                def unique_sessions = ses_list.findAll { it && it != '' }.unique()
                def session_count = unique_sessions.size()
                [sub, session_count]
            }

        // Step 1: Join anatomical image with segmentation
        def surf_recon_input_base = anat_for_surf_recon
            .join(anat_skull_seg.map { sub, ses, seg_file -> [sub, ses, seg_file] }, by: [0, 1], remainder: true)
            .map { sub, ses, anat_file, bids_name, seg_file ->
                [sub, ses, anat_file, bids_name, seg_file]
            }

        // Step 2: Join with brain mask
        def surf_recon_input_with_mask = surf_recon_input_base
            .join(anat_skull_mask.map { sub, ses, mask_file -> [sub, ses, mask_file] }, by: [0, 1], remainder: true)
            .map { sub, ses, anat_file, bids_name, seg_file, mask_file ->
                def final_mask = mask_file ?: file("${workDir}/dummy_brain_mask.dummy").tap { it.toFile().text = "" }
                [sub, ses, anat_file, bids_name, seg_file, final_mask]
            }

        // Step 3: Join with optional ARM6 atlas
        def surf_recon_input_with_arm6 = surf_recon_input_with_mask
            .join(anat_arm6_atlas.map { sub, ses, arm6_file -> [sub, ses, arm6_file] }, by: [0, 1], remainder: true)
            .map { sub, ses, anat_file, bids_name, seg_file, mask_file, arm6_file ->
                def final_arm6 = arm6_file ?: file("${workDir}/dummy_arm6_atlas.dummy").tap { it.toFile().text = "" }
                [sub, ses, anat_file, bids_name, seg_file, mask_file, final_arm6]
            }

        // Step 4: Join with session count
        def anat_sessions_clean = anat_sessions_per_subject
            .unique { sub, session_count -> sub }
            .map { sub, session_count -> [sub, session_count] }

        def surf_recon_input = surf_recon_input_with_arm6
            .combine(anat_sessions_clean, by: 0)
            .map { sub, ses, anat_file, bids_name, seg_file, mask_file, arm6_atlas_file, session_count ->
                def count = session_count instanceof List ? session_count[0] : session_count
                [sub, ses, anat_file, bids_name, seg_file, mask_file, arm6_atlas_file, count]
            }

        ANAT_SURFACE_RECONSTRUCTION(surf_recon_input, config_file)

        surf_actual_subject_id_ch = ANAT_SURFACE_RECONSTRUCTION.out.actual_subject_id
            .map { sub, ses, id_file -> [sub, ses, id_file.text.trim()] }

        surf_subject_dir_ch = ANAT_SURFACE_RECONSTRUCTION.out.subject_dir

        // Step 5: Prepare QC input channels
        def surf_qc_bids_lookup = anat_for_surf_recon
            .map { sub, ses, anat_file, bids_name ->
                [sub, ses, bids_name]
            }

        def surf_qc_input = ANAT_SURFACE_RECONSTRUCTION.out.subject_dir
            .join(ANAT_SURFACE_RECONSTRUCTION.out.actual_subject_id, by: [0, 1])
            .join(ANAT_SURFACE_RECONSTRUCTION.out.metadata, by: [0, 1])
            .join(surf_qc_bids_lookup, by: [0, 1])
            .map { sub, ses, subject_dir, actual_subject_id_file, metadata_file, bids_name ->
                def atlas_name = "ARM2"
                try {
                    def metadata = new groovy.json.JsonSlurper().parse(metadata_file)
                    atlas_name = metadata.atlas_name ?: "ARM2"
                } catch (Exception e) {
                    println "Warning: Could not read atlas_name from metadata, using default: ${e.message}"
                }
                def actual_subject_id = actual_subject_id_file.text.trim()
                [sub, ses, actual_subject_id, bids_name, atlas_name]
            }

        // Step 5: Run QC processes
        def surf_tissue_seg_qc_input = surf_qc_input
            .map { sub, ses, actual_subject_id, bids_name, atlas_name ->
                [sub, ses, actual_subject_id, bids_name]
            }
        QC_SURF_RECON_TISSUE_SEG(surf_tissue_seg_qc_input, config_file)
        QC_CORTICAL_SURF_AND_MEASURES(surf_qc_input, config_file)

        // Collect QC channels for completion signal
        surf_qc_channels = QC_SURF_RECON_TISSUE_SEG.out.metadata
            .mix(QC_CORTICAL_SURF_AND_MEASURES.out.metadata)

        // ============================================
        // LONGITUDINAL STREAM (synthesis_level: session_longitudinal)
        // ============================================
        // Shape change: the cross-sectional fan-out above is gathered to one
        // task per subject to build an unbiased base template, then fanned out
        // again so each session is reconstructed from that base.
        if (longitudinal_enabled) {

            // ---- GATHER: every session of a subject ----------------------
            // Plain join, not remainder: both sides come from the same process,
            // so a row exists in both or in neither. remainder:true would add
            // short tuples for nothing and mask a real emission bug.
            // How many sessions *should* contribute, derived fresh rather than
            // reusing anat_sessions_clean, which step 4 already consumed.
            def expected_sessions_per_subject = anat_for_surf_recon
                .map { sub, ses, anat_file, bids_name -> [sub, ses] }
                .unique()
                .groupTuple(by: 0)
                .map { sub, ses_list ->
                    [sub, ses_list.findAll { it && it != '' }.unique().size()]
                }

            def cross_per_session = ANAT_SURFACE_RECONSTRUCTION.out.base_inputs
                .join(ANAT_SURFACE_RECONSTRUCTION.out.actual_subject_id, by: [0, 1])
                // combine, not join: join is 1:1 and would consume the single
                // per-subject count row on the first session, silently dropping
                // every later session of that subject.
                .combine(expected_sessions_per_subject, by: 0)
                .map { sub, ses, staged_files, aid_file, session_count ->
                    def count = session_count instanceof List ? session_count[0] : session_count
                    [sub, ses, staged_files, aid_file.text.trim(), count]
                }

            // groupTuple without size: waits for the channel to close, so no
            // base starts until the slowest cross-sectional recon in the whole
            // cohort finishes. groupKey(sub, session_count) would fix that, but
            // ANAT_SURFACE_RECONSTRUCTION carries errorStrategy 'ignore': a
            // failed session emits nothing, and a key sized to session_count
            // would then never complete, deadlocking that subject's base
            // forever. Slower is better than stuck.
            def base_build_input = cross_per_session
                .groupTuple(by: 0)
                .filter { sub, ses_list, files_list, cross_ids, counts ->
                    // A base needs at least two timepoints. One-session
                    // subjects are skipped deliberately: FreeSurfer's 1-tp base
                    // exists only for cohort uniformity in group stats, and
                    // here it would add an interpolation for no gain.
                    if (cross_ids.size() < 2) {
                        println "Note: sub-${sub} has ${cross_ids.size()} session(s) with anatomy; skipping the longitudinal base template (needs >= 2)."
                        return false
                    }
                    return true
                }
                .map { sub, ses_list, files_list, cross_ids, counts ->
                    // Sort by session id. groupTuple emits in completion order,
                    // not input order, so without this the base-tps ordering,
                    // the LTA filenames and mri_robust_template's --inittp would
                    // all vary between runs, breaking -resume reproducibility.
                    // toList() before sort: (0..<n) is an IntRange, an immutable
                    // AbstractList, and Groovy's sort(Closure) mutates in place --
                    // so sorting the range directly throws
                    // UnsupportedOperationException and aborts the whole run. It
                    // only fires once groupTuple emits, i.e. after every
                    // cross-sectional reconstruction has finished, so the cost of
                    // getting this wrong is hours.
                    def order = (0..<ses_list.size()).toList().sort { a, b ->
                        ("${ses_list[a] ?: ''}") <=> ("${ses_list[b] ?: ''}")
                    }
                    def expected = counts instanceof List ? counts[0] : counts
                    [ sub,
                      order.collect { "${cross_ids[it]}" }.join(','),
                      expected,
                      order.collect { files_list[it] }.flatten() ]
                }

            // ---- BASE, STAGE 1/3: robust average (cpu) -------------------
            ANAT_SURFACE_BASE_TEMPLATE(base_build_input, config_file)

            // ---- BASE, STAGE 2/3: segment + backproject atlases (gpu) ----
            // Split out so the GPU token is held only for a CNN segmentation and
            // one template registration, not for the multi-hour reconstruction
            // that follows. Same gate as ANAT_SKULLSTRIPPING: without it this
            // would pull a token and run on GPU even in CPU mode.
            def use_base_gpu = params.use_gpu
            def base_gpu_input = use_base_gpu ? gpu_queue : Channel.value('none')

            def base_atlas_input = ANAT_SURFACE_BASE_TEMPLATE.out.base_dir
                .map { sub, base_id, base_dir -> [sub, base_id] }
                .join(ANAT_SURFACE_BASE_TEMPLATE.out.base_nii, by: 0)

            ANAT_SURFACE_BASE_ATLAS(base_atlas_input, config_file, base_gpu_input)

            // Return the token so the next GPU task can take the slot.
            if (use_base_gpu) {
                ANAT_SURFACE_BASE_ATLAS.out.gpu_token.subscribe { gpu_queue << it }
            }

            // ---- BASE, STAGE 3/3: reconstruct the surfaces (cpu) ---------
            def base_recon_input = ANAT_SURFACE_BASE_TEMPLATE.out.base_dir
                .join(ANAT_SURFACE_BASE_TEMPLATE.out.base_nii, by: 0)
                .join(ANAT_SURFACE_BASE_ATLAS.out.segmentation, by: 0)
                .join(ANAT_SURFACE_BASE_ATLAS.out.brain_mask, by: 0)
                .join(ANAT_SURFACE_BASE_ATLAS.out.arm6_atlas, by: 0)

            ANAT_SURFACE_BASE_RECON(base_recon_input, config_file)

            // The *reconstructed* base is what timepoints seed from, so this is
            // stage 3's output, while the transforms come from stage 1.
            surf_base_dir_ch = ANAT_SURFACE_BASE_RECON.out.base_dir
                .map { sub, base_id, base_dir -> [sub, base_dir] }
            surf_base_subject_id_ch = ANAT_SURFACE_BASE_RECON.out.base_dir
                .map { sub, base_id, base_dir -> [sub, base_id] }

            // ---- FAN OUT AGAIN: one longitudinal recon per session --------
            // Each session's own BIDS stem, which the timepoint restamps into
            // base space.
            def bids_by_subject = anat_for_surf_recon
                .map { sub, ses, anat_file, bids_name -> [sub, ses, bids_name] }

            // combine(by:0), NOT join(by:0), for the per-subject rows: join is 1:1
            // and would consume the single base row on the first match, silently
            // dropping every later session of that subject. combine broadcasts it
            // -- the same idiom already used for anat_sessions_clean above. The
            // per-session rows above it are joined, since those are 1:1.
            def long_input = ANAT_SURFACE_RECONSTRUCTION.out.subject_dir
                .join(ANAT_SURFACE_RECONSTRUCTION.out.actual_subject_id, by: [0, 1])
                .join(bids_by_subject, by: [0, 1])
                // The reconstructed base (stage 3) -- timepoints inherit its
                // surfaces, so a partial tree from stage 1 would be useless --
                // and the transforms from stage 1, which is where they are made.
                .combine(ANAT_SURFACE_BASE_RECON.out.base_dir, by: 0)
                .combine(ANAT_SURFACE_BASE_TEMPLATE.out.tp_to_base_ltas, by: 0)
                .map { sub, ses, cross_dir, aid_file, bids_name, base_id, base_dir, ltas ->
                    [sub, ses, cross_dir, aid_file.text.trim(),
                     base_dir, base_id, ltas, bids_name]
                }

            ANAT_SURFACE_RECONSTRUCTION_LONG(long_input, config_file)

            surf_long_subject_dir_ch = ANAT_SURFACE_RECONSTRUCTION_LONG.out.subject_dir
            surf_long_subject_id_ch = ANAT_SURFACE_RECONSTRUCTION_LONG.out.actual_subject_id
                .map { sub, ses, id_file -> [sub, ses, id_file.text.trim()] }

            // ---- QC for the base and the longitudinal timepoints ----------
            // Both QC processes locate a subject dir by name under
            // output_dir/fastsurfer, which is exactly where these publish, so
            // they need no change -- only new input rows.
            //
            // They derive the figure filename from the BIDS stem alone, and all
            // of a subject's figures land in one sub-<id>/figures directory, so
            // the filename is the only thing separating a session's
            // cross-sectional figure from its longitudinal one. Each stream is
            // therefore named through longitudinal_bids_name() by the process
            // that produces it. The stem is only ever parsed for a name --
            // create_bids_output_filename touches no filesystem -- so a
            // synthetic one is safe.
            // atlas_name read from metadata rather than hard-coded: with a
            // non-default anat.skullstripping_segmentation.atlas_name the cortical
            // QC would look for label/?h.aparc.ARM2atlas.mapped.annot and silently
            // emit no figure.
            //
            // From the RECON's metadata, not ANAT_SURFACE_BASE_ATLAS's. Those are
            // two different values that only coincide today because both land on
            // ARM2: the recon takes atlas_name from config and uses it to *name*
            // ?h.aparc.<X>atlas.mapped.{annot,stats}, while the atlas stage's comes
            // from apply_segmentation, i.e. whatever the segmentation checkpoint
            // reports. The rule -- already what the cross-sectional QC above does --
            // is to read it from the metadata of the run that wrote the files you
            // are about to open.
            def atlasNameFrom = { metadata_file ->
                try {
                    new groovy.json.JsonSlurper().parse(metadata_file).atlas_name ?: "ARM2"
                } catch (Exception e) {
                    println "Warning: could not read atlas_name from metadata, using ARM2: ${e.message}"
                    "ARM2"
                }
            }
            def base_atlas_name = ANAT_SURFACE_BASE_RECON.out.metadata
                .map { sub, metadata_file -> [sub, atlasNameFrom(metadata_file)] }
            def long_atlas_name = ANAT_SURFACE_RECONSTRUCTION_LONG.out.metadata
                .map { sub, ses, metadata_file -> [sub, ses, atlasNameFrom(metadata_file)] }

            // The stems come from the processes that own them, not from a regex
            // here. The regex version could not work: bids_name is an absolute
            // path (create_synthesized_bids_filename's session-level branch
            // returns one), so both `^`-anchored rules silently matched nothing
            // and every longitudinal QC figure was written to exactly the
            // cross-sectional filename -- same directory, publishDir
            // overwrite:false, figure dropped.
            def base_qc_input = ANAT_SURFACE_BASE_RECON.out.base_dir
                .map { sub, base_id, base_dir -> [sub, base_id] }
                .join(
                    ANAT_SURFACE_BASE_ATLAS.out.bids_name
                        .map { sub, name_file -> [sub, name_file.text.trim()] },
                    by: 0
                )
                .join(base_atlas_name, by: 0)
                .map { sub, base_id, base_stem, atlas_name ->
                    [sub, '', base_id, base_stem, atlas_name]
                }

            // .out directly, not surf_long_subject_id_ch: that one is also named
            // in emit:, and consuming a channel in an operator *and* emitting it
            // is the pattern that breaks the moment a parent workflow reads the
            // emit. Process outputs are multicast, so this is safe.
            def long_qc_input = ANAT_SURFACE_RECONSTRUCTION_LONG.out.actual_subject_id
                .map { sub, ses, id_file -> [sub, ses, id_file.text.trim()] }
                .join(
                    ANAT_SURFACE_RECONSTRUCTION_LONG.out.bids_name
                        .map { sub, ses, name_file -> [sub, ses, name_file.text.trim()] },
                    by: [0, 1]
                )
                .join(long_atlas_name, by: [0, 1])
                .map { sub, ses, long_id, long_stem, atlas_name ->
                    [sub, ses, long_id, long_stem, atlas_name]
                }

            def extra_tissue_qc = base_qc_input
                .mix(long_qc_input)
                .map { sub, ses, actual_subject_id, bids_name, atlas_name ->
                    [sub, ses, actual_subject_id, bids_name]
                }
            QC_SURF_RECON_TISSUE_SEG_LONG(extra_tissue_qc, config_file)
            QC_CORTICAL_SURF_AND_MEASURES_LONG(base_qc_input.mix(long_qc_input), config_file)

            surf_qc_channels = surf_qc_channels
                .mix(QC_SURF_RECON_TISSUE_SEG_LONG.out.metadata)
                .mix(QC_CORTICAL_SURF_AND_MEASURES_LONG.out.metadata)

            // ---- ATLASES ONTO THE BASE SURFACE ---------------------------
            // Projected onto the base only, and that is the right answer rather
            // than a compromise: every timepoint inherits the base's mesh, so a
            // vertex label assigned on the base *is* that vertex's label in all
            // of them. Per-timepoint projections would be near-duplicates.
            //
            // They would also collide. anat_project_atlases_to_surface names its
            // outputs through get_bids_prefix(), which keeps only sub/ses, so a
            // per-timepoint projection would land on exactly the cross-sectional
            // filenames in the same publish directory. The base is safe twice
            // over: it publishes to sub-X/anat/ while sessions publish to
            // sub-X/ses-Y/anat/, and its atlases carry space-base -- taken from
            // the base's own name, not from the stem get_bids_prefix builds.
            //
            // Gated on the atlas list actually being non-empty rather than on
            // registration_enabled/is_custom_template: this covers those cases
            // *and* a registration that produced no inverse transform, and it
            // needs no config resolution here.
            def base_atlas_files = ANAT_SURFACE_BASE_ATLAS.out.atlases
                .filter { sub, files ->
                    def list = files instanceof List ? files : (files ? [files] : [])
                    if (!list) {
                        println "Note: sub-${sub} has no backprojected atlases (custom template, or registration disabled); skipping fsnative atlas projection for its base."
                        return false
                    }
                    return true
                }

            def base_surf_atlas_input = base_qc_input
                .map { sub, ses, base_id, bids_name, atlas_name -> [sub, bids_name] }
                // .out directly, not surf_base_dir_ch: that one is also named in
                // emit:, and consuming a channel in an operator *and* emitting it
                // is the pattern that breaks the moment a parent workflow reads
                // the emit. Process outputs are multicast, so this is safe.
                .join(
                    ANAT_SURFACE_BASE_RECON.out.base_dir
                        .map { sub, base_id, base_dir -> [sub, base_dir] },
                    by: 0
                )
                .combine(base_atlas_files, by: 0)
                .map { sub, bids_name, base_dir, atlas_files ->
                    [sub, '', bids_name, atlas_files, base_dir]
                }
            ANAT_PROJECT_ATLASES_TO_SURFACE_BASE(base_surf_atlas_input, config_file)

            // ---- CHANGE STATISTICS: one task per subject ------------------
            // Gathers all of a subject's longitudinal timepoints, so it runs
            // once they have all finished. groupTuple again without size:, for
            // the same errorStrategy 'ignore' reason as the base gather.
            def change_stats_input = ANAT_SURFACE_RECONSTRUCTION_LONG.out.subject_dir
                .join(ANAT_SURFACE_RECONSTRUCTION_LONG.out.actual_subject_id, by: [0, 1])
                .map { sub, ses, long_dir, id_file -> [sub, ses, long_dir, id_file.text.trim()] }
                .groupTuple(by: 0)
                .filter { sub, ses_list, long_dirs, long_ids ->
                    if (long_ids.size() < 2) {
                        println "Note: sub-${sub} has ${long_ids.size()} longitudinal timepoint(s); skipping change statistics (needs >= 2)."
                        return false
                    }
                    return true
                }
                .map { sub, ses_list, long_dirs, long_ids ->
                    // Sorted for the same reproducibility reason as the base gather.
                    // toList() before sort: (0..<n) is an IntRange, an immutable
                    // AbstractList, and Groovy's sort(Closure) mutates in place --
                    // so sorting the range directly throws
                    // UnsupportedOperationException and aborts the whole run. It
                    // only fires once groupTuple emits, i.e. after every
                    // cross-sectional reconstruction has finished, so the cost of
                    // getting this wrong is hours.
                    def order = (0..<ses_list.size()).toList().sort { a, b ->
                        ("${ses_list[a] ?: ''}") <=> ("${ses_list[b] ?: ''}")
                    }
                    [ sub,
                      order.collect { "${long_ids[it]}" }.join(','),
                      order.collect { long_dirs[it] } ]
                }
                // From .out directly rather than the surf_base_* channels, which
                // the QC rows above already consume; process outputs are
                // multicast, so reusing them is safe.
                .combine(
                    ANAT_SURFACE_BASE_RECON.out.base_dir
                        .map { sub, base_id, base_dir -> [sub, base_dir, base_id] },
                    by: 0
                )
                .combine(base_atlas_name, by: 0)
                // The subject's own sessions.tsv, which is where BIDS keeps the
                // real elapsed time. Without it the fitted rate is per scan, or
                // keyed to whatever digits lead the session label -- a different
                // quantity wearing the same name. A .dummy placeholder keeps the
                // input arity fixed when the dataset has none; the process tests
                // the filename, so the placeholder must not end in _sessions.tsv.
                .map { sub, long_ids_csv, long_dirs, base_dir, base_id, atlas_name ->
                    def tsv = file("${params.bids_dir}/sub-${sub}/sub-${sub}_sessions.tsv")
                    def sessions_tsv = tsv.exists()
                        ? tsv
                        : file("${workDir}/no_sessions_tsv.dummy").tap { it.toFile().text = "" }
                    [sub, long_ids_csv, long_dirs, base_dir, base_id, atlas_name, sessions_tsv]
                }

            ANAT_SURFACE_LONG_CHANGE_STATS(change_stats_input, config_file)

            surf_qc_channels = surf_qc_channels
                .mix(ANAT_SURFACE_LONG_CHANGE_STATS.out.metadata)
        }
    } else {
        if (surf_recon_enabled && !anat_skullstripping_enabled) {
            println "Warning: Surface reconstruction is enabled but skullstripping is disabled. Skipping surface reconstruction."
        }
        if (longitudinal_enabled) {
            println "Warning: anat.synthesis_level is 'session_longitudinal' but surface reconstruction is not running, so no base template or longitudinal reconstruction will be produced."
        }
        // Emit single value so main.nf QC completion doesn't hang when surf recon is skipped
        surf_qc_channels = Channel.value('skipped')
    }

    // ============================================
    // EMIT OUTPUT CHANNELS
    // ============================================
    // Explicit name required for sub-workflow output access (Nextflow 25.x)
    emit:
    surf_qc_channels
    surf_actual_subject_id_ch
    surf_subject_dir_ch
    surf_base_dir_ch
    surf_base_subject_id_ch
    surf_long_subject_dir_ch
    surf_long_subject_id_ch
}
