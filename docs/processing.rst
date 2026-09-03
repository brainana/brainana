:og:description: Detailed methods for every Brainana macaque MRI preprocessing stage: anatomical synthesis and segmentation, cortical surface reconstruction, and functional (fMRI/BOLD) preprocessing.

.. meta::
   :description: Detailed methods for every Brainana macaque MRI preprocessing stage: anatomical synthesis and segmentation, cortical surface reconstruction, and functional (fMRI/BOLD) preprocessing.
   :keywords: macaque MRI preprocessing, anatomical processing, surface reconstruction, fMRI preprocessing, FreeSurfer, segmentation

Processing details
==================

Brainana adapts its pipeline depending on what data and metadata are
available and on the configuration you provide. For example,
anatomical synthesis runs only when multiple T1w/T2w runs or sessions
are present and synthesis is enabled; slice timing correction runs
only when slice timing information is available in the BIDS metadata.

This page describes the methods used at each stage of the pipeline and
links to related anatomical and space-tracking details.


1. BIDS discovery and job creation
----------------------------------

Before the Nextflow workflow runs, a Python discovery step scans the
BIDS dataset and produces structured job descriptors.

- **Purpose:** Determine what data are available and which processing
  branches should run (e.g. whether anatomical synthesis is needed,
  whether functional runs have slice timing metadata).
- **Methods:** Deterministic layout/metadata scan using BIDS entities
  and configuration. Discovery evaluates whether anatomical synthesis
  is needed and decides synthesis type/level
  (session vs. subject) from config. There are **no imaging
  algorithms** at this stage.


2. Anatomical processing
------------------------

.. figure:: _static/pipeline_details/anat_workflow.png
   :alt: Overview of anatomical preprocessing workflow.
   :align: center
   :width: 100%

|

The anatomical branch turns raw (or synthesized) T1w/T2w images into
bias-corrected, skull-stripped, and template-registered images, and
optionally tissue segmentations. These outputs provide the anatomical
reference for T2w, functional, and surface workflows.


2.1 Ingest normalization and synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All anatomicals pass through ingest normalization before later steps;
optional synthesis runs afterward when multiple T1w/T2w runs exist.

- **Purpose:** Fix NIfTI header issues with one unambiguous fix, flag
  ambiguous ones, and optionally merge multiple anatomicals into one
  reference.
- **Pass-through:** One read/write pass; well-formed inputs are not
  rewritten (byte-identical). 

**Automatic header repairs** (metadata only — no resampling or
reorientation). Sidecar keys are set when a repair runs; otherwise
omitted. Rewriting can re-encode ``scl_slope`` / ``scl_inter`` integers
by one quantization step; scaled values are unchanged.

- **4D shape** (``Input4DCollapsed``): drop a trailing singleton
  dimension, or average a true multi-volume anatomical on the last axis.
- **Missing qform/sform** (``OrientationRecovered``): both codes 0 leaves
  readers disagreeing (nibabel/FSL vs ITK/ANTs). Brainana writes the
  nibabel/FSL fallback to qform and sform (code 2).
- **Conflicting qform/sform** (``QformSformReconciled``): sform is
  copied to both (brainana/FSL use sform; FastSurfer favors qform).
- **``.nii`` → ``.nii.gz``:** converted once at ingest (not logged in
  the sidecar).

.. warning::

   ``OrientationRecovered`` assigns a *convention*, not verified
   handedness. Wrong acquisition direction can yield an invisible
   left/right mirror. Confirm against external records before
   hemisphere-level interpretation.

**Reported only** (``InputHeaderWarnings`` in sidecar and QC **Data
findings**; section omitted if empty):

- Voxel sizes not in mm (nibabel vs ITK/ANTs interpret ``xyzt_units``
  differently).
- ``pixdim`` inconsistent with the affine (surfaced at ingest instead
  of failing deep in segmentation).

**Synthesis (conditional)**

- **When:** Multiple T1w or T2w per session or subject (configurable
  level).
- **Method:** Lexicographically first image is the fixed reference;
  other images are rigidly registered with ANTs, then all runs are
  averaged in reference space.


2.2 Conform to reference
~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Align the brain to reference space (orientation and
  grid) so that subsequent registrations and resamplings are
  well-defined.
- **Method:**

  1. Skull-strip the input with a UNet-based skull-stripping model, or
     use an existing brain mask when skull stripping is disabled.
  2. Resample the template to match the input resolution if needed.
  3. Run FSL FLIRT (rigid, 6 DOF) from the brain-extracted input to
     the template.
  4. Use AFNI ``3dresample`` to ensure template and input share a
     consistent grid.
  5. Apply the FLIRT transform back to the full-head anatomical.

- **Field of view:** The output grid is the template's box, so it is
  sized for a brain rather than a head, and anything outside it is
  cropped. All downstream steps use that cropped image. Alongside it,
  the step also writes ``desc-conformFullFOV`` — the same conform on a
  grid enlarged just enough to contain every voxel of the input — for
  users who need what falls outside, such as a recording chamber or
  head-post. Nothing downstream reads it. The QC report shows both
  fields of view: the uncropped one with the processing box drawn on
  it, and that box enlarged.


2.3 Skullstripping and segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Provide brain mask and atlas/tissue segmentation for
  masking, bias correction, registration, and optional surface
  reconstruction.
- **Method:** A FastSurfer-style CNN segmentation model, fine-tuned on
  macaque anatomical MRI with CHARM and SARM level 2 atlases (ARM2
  parcellation). The network produces an atlas-labelled segmentation,
  from which a brain mask and optional hemisphere masks are derived.


2.4 Bias field correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Correct intensity non-uniformity (INU) in anatomical
  images.
- **Method:** N4 bias field correction (ANTs
  ``N4BiasFieldCorrection``) is run on the anatomical image, with the
  brain mask from segmentation optionally provided to
  restrict correction to brain tissue.


2.5 Registration to template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Map anatomical data to a standard template space (for
  example, NMT2Sym).
- **Method:** Multi-stage ANTs registration:

  - Translation → rigid → affine → optional SyN.
  - Metrics (e.g. mutual information, cross-correlation, Mattes),
    gradient steps, shrink factors, convergence criteria, and
    smoothing schedules are configurable.
  - When GPU resources and FireANTs are available, the SyN stage can
    be run with FireANTs.


2.6 T2w to T1w coregistration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When T2w data are present, Brainana can coregister T2w to the
preprocessed T1w:

- **Method:** ANTs rigid registration from T2w to preprocessed T1w.


3. Surface reconstruction
-------------------------

.. figure:: _static/pipeline_details/surf_workflow.png
   :alt: Overview of surface reconstruction workflow.
   :align: center
   :width: 100%

|


Surface reconstruction runs after anatomical preprocessing when enabled.
It is an optional, compute-intensive step that extends segmentation
outputs to build FreeSurfer-compatible cortical meshes and morphometric
maps.

- **When:** Controlled by ``anat.surface_reconstruction.enabled`` (enabled
  by default). Requires a valid FreeSurfer license (see
  :ref:`the-freesurfer-license-optional`).
- **Purpose:** Reconstruct white-matter and pial cortical surfaces and
  derive morphological measures (e.g. cortical thickness, surface area,
  curvature).
- **Inputs:** Preprocessed T1w, ARM2 atlas segmentation, and brain mask
  from section 2 (skull stripping and segmentation).
- **Method:** A FastSurfer-style workflow built on FreeSurfer, with
  macaque-specific adaptations:

  - Convert CNN-derived ARM2 labels into a FreeSurfer-compatible segmentation.
  - Tune surface reconstruction parameters for submillimeter macaque MRI.
  - Apply targeted segmentation refinements in error-prone regions,
    including the occipital calcarine cortex, claustrum, and
    orbitofrontal cortex.
  - Run an additional topology correction step for surface defects that
    commonly arise in macaque reconstructions and are not reliably
    resolved by FreeSurfer alone.

- **Outputs:** FreeSurfer-compatible subject directories under ``fastsurfer/``.


4. Functional processing
------------------------

.. figure:: _static/pipeline_details/func_workflow.png
   :alt: Overview of functional preprocessing workflow.
   :align: center
   :width: 100%

|


The functional branch preprocesses fMRI data and produces
motion-corrected, optionally slice-time-corrected, despiked,
bias-corrected, and skullstripped fMRI in native or template space,
with associated transforms and QC outputs.

The workflow is conceptually split into:

- **Time-series steps:** Slice timing (if available) →
  motion correction and temporal mean → 
  within-session coregistration and session-averaged temporal mean.
- **Compute on temporal mean:** Bias correction → conform → brain mask
  (UNet) → registration to anatomical or template.
- **Apply to 4D:** Apply conform and registration transforms to the
  4D fMRI and brain mask.


4.1 Slice timing correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When slice timing information is available in BIDS metadata, Brainana
applies slice timing correction.

- **Purpose:** Align voxel time series in time according to the slice
  acquisition order.
- **Method:** AFNI ``3dTshift`` is used to shift slices in time to a
  reference (typically the middle of the TR). Slice timing pattern
  (e.g. ``alt+z``, ``seq+z``) is derived from the BIDS
  ``SliceTiming`` and ``SliceEncodingDirection`` fields. If slice
  encoding is not along the z axis, data are swapped to z for
  ``3dTshift`` and swapped back afterwards.
- **Control:** This step can be disabled in configuration or skipped
  when slice timing metadata are missing.


4.2 Motion correction
~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Realign fMRI volumes to correct for subject motion.
- **Method:** FSL ``mcflirt`` performs volume realignment with 6 DOF.
  The reference volume is either a user-specified timepoint, the
  middle volume, or a temporal mean (via ``fslmaths -Tmean`` or
  ``fslroi``). Outputs include motion-corrected 4D fMRI, motion
  matrices, and motion parameters (TSV).
- **Short runs:** For very short runs (e.g. fewer than 15 volumes),
  motion correction can be skipped, and pass-through outputs
  (including a temporal mean and zero-filled motion parameters) are
  generated.


4.3 Despike
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **When:** Optional; controlled by ``func.despike.enabled`` (**off by
  default**). Can ignore the first N volumes via
  ``func.despike.ignore_first_volumes``.
- **Purpose:** Remove transient intensity spikes from the fMRI time
  series.
- **Method:** AFNI ``3dDespike`` with local editing (cutoff parameters
  ``c1``/``c2``, default 5/10).


4.4 Within-session coregistration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple fMRI runs exist per session, an optional within-session
coregistration step can align runs to a common reference and produce
a session-averaged temporal mean.

- **Purpose:** Improve stability of the temporal mean used for
  bias correction, conform, and registration.
- **Method:** Rigid registration from each run's temporal mean to a
  reference run's temporal mean, using ANTs or FSL FLIRT as
  configured, followed by applying the transform to the 4D fMRI and
  mask.


4.5 Bias correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Improve downstream registration by correcting low frequency 
  intensity inhomogeneity in the reference temporal mean fMRI image only. 
  Bias correction is not applied to the full 4D timeseries, preserving 
  the original temporal signal for subsequent analyses.
- **Method:** N4 bias field correction (ANTs
  ``N4BiasFieldCorrection``) is applied to the temporal mean of the
  motion-corrected fMRI (and to anatomical images in the anatomical
  workflow). Before N4, intensities may be rescaled to a non-zero mean
  of 100 (configurable), and any negative voxels are clamped to zero
  with a warning (N4 uses log-domain math). Anatomical bias correction
  uses a brain mask from skull stripping so background zeros are
  excluded from the histogram; functional bias correction runs earlier
  in the pipeline (before conform/skull strip) and therefore has no
  brain mask at this step—background zeros may remain in the image.


4.6 Conform and skull stripping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Conform:** The functional temporal mean is conformed to the
  chosen template using the same strategy as anatomical conform:
  skull-stripping the mean, FSL FLIRT rigid registration to the
  template, resampling with AFNI ``3dresample``, and application of the
  transform (optionally composed with anatomical transforms).
- **Skull stripping:** A UNet-based functional (EPI) skull-stripping
  model, derived from NHP-BrainExtraction/DeepBet, is run on the
  temporal mean to obtain a brain mask, which is then applied to the
  4D fMRI.


4.7 Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Method:** ANTs registration (rigid, affine, or SyN as configured)
  from the mean functional image to the preprocessed anatomical or
  directly to the template. Composite transforms are applied to the
  4D fMRI and brain mask with ``antsApplyTransforms`` (e.g. BSpline
  interpolation for fMRI).
- **Outputs:** Preprocessed fMRI and mask in anatomical or template
  space.


4.8 Temporal SNR (tSNR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose:** Provide a quick functional-quality map for QC.
- **Method:** Voxelwise temporal SNR (``|mean| / SD`` over time) of the
  preprocessed 4D fMRI, saved as per-run and session-average maps
  (``*_stat-tsnr_boldmap``) with optional projection to the surface.
- **Control:** Skipped for runs that are not 4D or have fewer than 10
  timepoints.


4.9 Confound regressors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Brainana estimates fMRIPrep-compatible nuisance regressors for optional
downstream denoising. These are **regressors only** — the fMRI image is
never scrubbed or modified.

- **When:** Controlled by ``func.confounds.enabled`` (**on by default**);
  runs after registration whenever the required inputs are available.
  Registration must be enabled, since the regressors are computed in the
  registered BOLD space. The motion-derived columns (24-parameter motion,
  ``framewise_displacement``, ``rmsd``) are produced only when
  ``func.motion_correction.enabled`` is also true; with motion correction
  off, those columns are omitted and the remaining regressors
  (``dvars``/``std_dvars``, ``global_signal``, tissue, DVARS-based
  outliers) are still computed. Three settings are configurable:
  ``func.confounds.fd_outlier_threshold_mm`` (default 0.25),
  ``func.confounds.std_dvars_outlier_threshold`` (default 1.5) and
  ``func.confounds.fd_radius_mm`` (default 27.0), the head radius used to
  convert rotations to millimetres.
- **Inputs:** Preprocessed 4D fMRI (T1w space preferred), motion
  parameters (when motion correction is enabled), and a brain mask
  (required for DVARS and global signal). Tissue regressors additionally
  require a T1w segmentation and its label lookup table.
- **Method:** A dependency-light reimplementation (NumPy/pandas/nibabel)
  of the standard Power/Jenkinson/nipype formulas — not nipype itself.
  Output is compatible with
  ``nilearn.interfaces.fmriprep.load_confounds``. CompCor and cosine
  regressors are **not** produced.
- **Regressors:**

  - 24-parameter motion: ``trans_x/y/z`` and ``rot_x/y/z`` with their
    derivatives and squared terms.
  - ``framewise_displacement`` (Power et al. 2012) and ``rmsd``
    (Jenkinson 1999). Both convert rotations to millimetres on a sphere of
    radius ``func.confounds.fd_radius_mm``, which defaults to the macaque
    27 mm; set it to the species' own head radius for other primates. FD
    values — and therefore ``fd_outlier_threshold_mm`` — are only comparable
    between runs computed at the same radius, so the radius actually used is
    recorded in the JSON sidecar.
  - ``dvars`` and ``std_dvars``.
  - ``global_signal`` (with expansions); ``csf``, ``white_matter`` and
    ``csf_wm`` are added **only when a T1w segmentation is available**.
  - Outlier indicators: ``non_steady_state_outlier##`` and
    ``motion_outlier##``.
- **Outputs:** BIDS ``*_desc-confounds_timeseries.tsv`` plus a JSON
  sidecar, and an fMRIPrep-style confounds panel in the QC report. The
  confounds and motion QC figures shade the outlier frames as vertical
  bands — gray for ``non_steady_state_outlier##``, red for
  ``motion_outlier##``. See :doc:`outputs` for the full column list.


5. Summary table
----------------

.. list-table::
   :header-rows: 1

   * - Domain
     - Step
     - Main tool / method
   * - Anatomical
     - Ingest normalization
     - nibabel header repair (4D collapse, missing qform/sform)
   * - Anatomical
     - Synthesis
     - ANTs rigid + average
   * - Anatomical
     - Conform
     - FLIRT + UNet skull-strip + 3dresample
   * - Anatomical
     - Skull strip & segment
     - FastSurfer-style CNN (fine-tuned)
   * - Anatomical
     - Bias correction
     - ANTs N4BiasFieldCorrection
   * - Anatomical
     - Registration
     - ANTs (optional FireANTs for SyN)
   * - Surface
     - Surface recon
     - FastSurfer-style workflow + FreeSurfer
   * - Functional
     - Slice timing
     - AFNI 3dTshift
   * - Functional
     - Motion correction
     - FSL mcflirt
   * - Functional
     - Despike
     - AFNI 3dDespike (optional)
   * - Functional
     - Within-session coreg
     - ANTs or FLIRT
   * - Functional
     - Bias correction
     - ANTs N4BiasFieldCorrection
   * - Functional
     - Conform / skull strip
     - FLIRT + UNet skull-strip + 3dresample
   * - Functional
     - Registration
     - ANTs (optional FireANTs for SyN)
   * - Functional
     - tSNR
     - nibabel (``|mean| / SD`` over time)
   * - Functional
     - Confounds
     - Custom fMRIPrep-style regressors

For outputs and directory layout, see :doc:`outputs`.

.. seealso::

   - :doc:`anat_selection_for_func` — how the T1w reference for fMRI is selected
   - :doc:`spaces_and_transforms` — spaces and transforms

