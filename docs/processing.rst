:og:description: Detailed methods for every Brainana macaque MRI preprocessing stage: anatomical synthesis and segmentation, cortical surface reconstruction, and functional (fMRI/BOLD) preprocessing.

.. meta::
   :description: Detailed methods for every Brainana macaque MRI preprocessing stage: anatomical synthesis and segmentation, cortical surface reconstruction, and functional (fMRI/BOLD) preprocessing.
   :keywords: macaque MRI preprocessing, anatomical processing, surface reconstruction, fMRI preprocessing, FreeSurfer, segmentation

Processing details
==================

Brainana adapts its pipeline depending on what data and metadata are
available and on the configuration you provide. For example,
anatomical synthesis runs only when there are several T1w/T2w runs to
combine at the configured ``anat.synthesis_level``; slice timing correction runs
only when slice timing information is available in the BIDS metadata.

This page describes the methods used at each stage of the pipeline and
links to related anatomical and space-tracking details.


BIDS discovery and job creation
-------------------------------

Before the Nextflow workflow runs, a Python discovery step scans the
BIDS dataset and produces structured job descriptors. It determines what
data are available and which processing branches should run, for example
whether anatomical synthesis is needed and whether functional runs have
slice timing metadata.

The scan is deterministic and uses only BIDS entities, metadata and the
configuration. Discovery decides whether anatomical synthesis is needed and
picks the synthesis type and level (session or subject) from the config. No
imaging algorithms run at this stage.

Discovery also checks that file names and directories agree on subject and
session, and stops with a report naming the files when they do not. T1w/T2w
files labeled ``part-phase``, ``part-real`` or ``part-imag`` are listed as not
processed and are not averaged into synthesis.


Anatomical processing
---------------------

The anatomical branch turns raw (or synthesized) T1w/T2w images into
bias-corrected, skull-stripped, and template-registered images, and
optionally tissue segmentations. These outputs provide the anatomical
reference for T2w, functional, and surface workflows.

.. figure:: _static/pipeline_details/anat_workflow.png
   :alt: Overview of anatomical preprocessing workflow.
   :align: center
   :width: 100%

   Anatomical workflow.


Ingest normalization and synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All anatomicals pass through ingest normalization before later steps.
Ingest fixes NIfTI header issues that have one unambiguous fix, flags
ambiguous ones, and can merge several anatomicals into one reference. It
makes one read/write pass, and well-formed inputs are not rewritten
(they stay byte-identical).

Header repairs change metadata only; images are never resampled or
reoriented. Each repair sets a sidecar key when it runs; otherwise the key
is omitted. Rewriting can re-encode ``scl_slope`` / ``scl_inter`` integers
by one quantization step; scaled values are unchanged. The repairs are:

4D shape (``Input4DCollapsed``)
   Drop a trailing singleton dimension, or average a true multi-volume
   anatomical on the last axis.

Missing qform/sform (``OrientationRecovered``)
   With both codes 0, readers disagree (nibabel/FSL vs ITK/ANTs). Brainana
   writes the nibabel/FSL fallback to qform and sform (code 2).

Conflicting qform/sform (``QformSformReconciled``)
   The sform is copied to both (Brainana and FSL use the sform; FastSurfer
   favors the qform).

NIfTI compression
   ``.nii`` inputs are converted to ``.nii.gz`` once at ingest (not logged
   in the sidecar).

.. warning::

   ``OrientationRecovered`` assigns a *convention*, not verified handedness.
   A wrong acquisition direction can yield an invisible left/right mirror.
   Confirm against external records before hemisphere-level interpretation.

Some issues are reported but not repaired. They are listed under
``InputHeaderWarnings`` in the sidecar and in the QC report's
:guilabel:`Data findings` section, which is omitted when empty:

- Voxel sizes not in mm (nibabel and ITK/ANTs interpret ``xyzt_units``
  differently).
- ``pixdim`` inconsistent with the affine (surfaced at ingest instead
  of failing deep in segmentation).

Synthesis runs afterward, and only when a session or subject has more than
one T1w or T2w (the level is configurable). The lexicographically first
image is the fixed reference; the other images are rigidly registered to it
with ANTs, and then all runs are averaged in reference space.


Conform to reference
~~~~~~~~~~~~~~~~~~~~

Conform aligns the brain to reference space (orientation and grid) so that
subsequent registrations and resamplings are well-defined:

1. Skull-strip the input with a UNet-based skull stripping model, or,
   when skull stripping is disabled, assume the input is already
   skull-stripped. If the model marks more than one separate region,
   the one it is most confident in is kept.
2. Resample the template to match the input resolution if needed.
3. Run FSL FLIRT (rigid, 6 DOF) from the brain-extracted input,
   cropped to the brain plus a margin, to the template. The crop
   keeps scans that include the neck or body from misaligning.
4. Use AFNI ``3dresample`` to ensure template and input share a
   consistent grid.
5. Apply the FLIRT transform back to the full-head anatomical.

The output grid is the template's box, so it is sized for a brain rather
than a head, and anything outside it is cropped. All downstream steps use
that cropped image. Alongside it, the step also writes
``desc-conformFullFOV`` — the same conform on a grid enlarged just enough to
contain every voxel of the input, up to 512 voxels per axis (a longer axis
keeps its central 512) — for users who need what falls outside, such as a
recording chamber or head-post. Nothing downstream reads it. The QC report
shows both fields of view: the uncropped one with the processing box drawn
on it, and that box enlarged.


.. _skullstripping-and-segmentation:

Skull stripping and segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step provides the brain mask and atlas/tissue segmentation used for
masking, bias field correction, registration, and optional surface
reconstruction. It uses a FastSurfer-style CNN segmentation model,
fine-tuned on macaque anatomical MRI with CHARM and SARM level 2 atlases
(ARM2 parcellation). The network produces an atlas-labeled segmentation,
from which a brain mask and optional hemisphere masks are derived.

Intensity scaling
   Intensities are rescaled to 0–255 before segmentation, capped relative
   to brain intensity so that very bright tissue outside the brain (e.g.
   next to a surface coil) does not compress the brain's range.

Second pass when brain is missed
   Strong intensity non-uniformity (a surface coil or a strong bias field)
   can make the network leave out dark brain. After the first pass, the
   NMT2Sym template is registered to the subject; if it shows brain outside
   the mask, the image is N4-corrected and segmented again, and the second
   result is kept only if it agrees with the template at least as well. If
   any of these checks fails, the first pass is kept. Settings:
   ``fastSurferCNN.pre_inference_n4`` and ``fastSurferCNN.template_prior``.

Other output spaces
   These template checks always use NMT2Sym. When the output space is
   another template, a custom template, or conform is off, the image is
   first rigidly aligned to NMT2Sym (a few seconds at most); this is
   automatic.

Provenance
   The brain-mask and segmentation sidecars record the mask volume, how
   many passes ran and why (``SegmentationPasses``), and the template
   comparison (``TemplatePrior``). A mask much smaller than the template
   brain is logged as a warning (``MaskUndersized``).

No brain found
   A mask under 10% of the template brain stops that subject:
   ``ANAT_SKULLSTRIPPING`` is listed as failed in the QC report, and the
   rest of the run continues. Check the subject's conform QC figure first;
   a conform that cropped to background is the usual cause.

Label fragments (optional)
   Detached label pieces smaller than
   ``fastSurferCNN.label_island_min_volume_mm3`` take the surrounding
   label. Off by default; 2.5 mm³ is a sensible value.


Bias field correction
~~~~~~~~~~~~~~~~~~~~~

Bias field correction removes intensity non-uniformity (INU) from
anatomical images. N4 bias field correction (ANTs ``N4BiasFieldCorrection``)
is run on the anatomical image, with the brain mask from segmentation
optionally provided to restrict correction to brain tissue.


Registration to template
~~~~~~~~~~~~~~~~~~~~~~~~

Registration maps anatomical data to a standard template space (for
example, NMT2Sym) with a multi-stage ANTs registration:

- Translation, then rigid, affine and SyN, up to the transform type set by
  ``registration.anat2template_xfm_type`` (default ``syn``). Stage
  metrics, iterations, shrink factors and smoothing are fixed; the
  transform type and ``registration.interpolation`` are configurable.
- With ``registration.enable_fireants: true`` (default), the SyN stage
  runs with FireANTs (affine followed by greedy deformable
  registration), on the GPU when one is assigned to the step and
  otherwise on the CPU. If FireANTs fails, ANTs is used.


T2w to T1w coregistration
~~~~~~~~~~~~~~~~~~~~~~~~~

When T2w data are present, Brainana can coregister T2w to the
preprocessed T1w with an ANTs rigid registration.


Surface reconstruction
----------------------

Surface reconstruction runs after anatomical preprocessing when enabled.
It is an optional, compute-intensive step that extends segmentation
outputs to build FreeSurfer-compatible cortical meshes and morphometric
maps: white-matter and pial cortical surfaces, and morphological measures
such as cortical thickness, surface area and curvature.

.. figure:: _static/pipeline_details/surf_workflow.png
   :alt: Overview of surface reconstruction workflow.
   :align: center
   :width: 100%

   Surface reconstruction workflow.

The step is controlled by ``anat.surface_reconstruction.enabled`` (enabled
by default) and requires a valid FreeSurfer license (see
:ref:`the-freesurfer-license-optional`). Its inputs are the preprocessed T1w,
the ARM2 atlas segmentation, and the brain mask from
:ref:`skullstripping-and-segmentation`.

The method is a FastSurfer-style workflow built on FreeSurfer, with
macaque-specific adaptations:

- Convert CNN-derived ARM2 labels into a FreeSurfer-compatible segmentation.
- Tune surface reconstruction parameters for submillimeter macaque MRI.
- Apply targeted segmentation refinements in error-prone regions,
  including the claustrum and orbitofrontal cortex (skipped with a custom
  template).

Starting surface and related settings:

Starting surface: the NMT2Sym template
   The default (``anat.surface_reconstruction.template_surface.enabled:
   true``). The NMT2Sym white surface is warped into the subject and fitted
   to the image, except in V1, where the white surface stays at the
   template: V1 has too little gray/white contrast on T1w images to place it
   reliably. The pial surface is fitted everywhere. Every subject shares the
   template's mesh and vertex numbering. If the template cannot be aligned
   to the subject, that subject is tessellated instead.
   ``scripts/?h.template_init.json`` and ``label/?h.template.V1.label``
   record what was used and held.

Starting surface: tessellation
   With ``template_surface.enabled: false`` (the 3.0.0 behavior), the
   white-matter segmentation is tessellated and its topology corrected with
   FreeSurfer (``mris_sphere -q``, ``mris_fix_topology``), with extra repair
   steps for defects common in macaque data.

``fix_V1_WM``
   ``fastSurferCNN.fix_V1_WM`` (default ``"auto"``) fills thin V1 white
   matter from the template before tessellation. ``"auto"`` turns it on
   only for tessellated surfaces.

Longitudinal
   Timepoints inherit the base's mesh and V1 label.

Surface QC record
   ``scripts/surface_qc.json`` records the surfaces' topology and the
   median cortical thickness per hemisphere; a median below 1.0 mm is
   flagged ``collapsed`` and logged (the pial surface most likely failed).

Outputs are FreeSurfer-compatible subject directories under ``fastsurfer/``.


Longitudinal stream (``session_longitudinal``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting ``anat.synthesis_level: "session_longitudinal"`` adds a second pass on
top of the per-session reconstruction above: an unbiased within-subject base
template, plus a base-seeded reconstruction per session, so every session of a
subject shares one cortical mesh and thickness can be differenced vertex by
vertex across timepoints.

This level also requires ``anat.surface_reconstruction.enabled``. Subjects
with fewer than two sessions carrying anatomy are skipped. The stream writes
``fastsurfer/sub-<id>_base/`` and ``fastsurfer/sub-<id>_ses-<id>_long/``, plus
per-vertex and per-ROI change statistics in the base directory. Functional
processing and the fsnative atlas outputs continue to use the
cross-sectional reconstructions.

The two phases and the time variable behind the change statistics are
described in :doc:`synthesis_level`.


Functional processing
---------------------

The functional branch preprocesses BOLD data and produces
motion-corrected and optionally slice-time-corrected and despiked BOLD
series in native or template space, with a brain mask, transforms and QC
outputs. Bias field correction and skull stripping are computed on the
temporal mean to guide registration; they are not applied to the 4D
series.

.. figure:: _static/pipeline_details/func_workflow.png
   :alt: Overview of functional preprocessing workflow.
   :align: center
   :width: 100%

   Functional workflow.

The workflow is conceptually split into three parts:

Time-series steps
   Slice timing (if available), then motion correction and temporal mean,
   despike (optional), and finally within-session coregistration and the
   session-averaged temporal mean.

Compute on temporal mean
   Bias field correction, then conform, brain mask (UNet), and registration
   to the anatomical or template.

Apply to 4D
   Apply the conform and registration transforms to the 4D BOLD series and
   brain mask.


Slice timing correction
~~~~~~~~~~~~~~~~~~~~~~~

When slice timing information is available in BIDS metadata, Brainana
applies slice timing correction to align voxel time series in time
according to the slice acquisition order.

AFNI ``3dTshift`` shifts slices in time to a reference (typically the
middle of the TR). The slice timing pattern (e.g. ``alt+z``, ``seq+z``) is
derived from the BIDS ``SliceTiming`` and ``SliceEncodingDirection`` fields.
If slice encoding is not along the z axis, data are swapped to z for
``3dTshift`` and swapped back afterwards.

This step can be disabled in configuration, and is skipped when slice
timing metadata are missing.


Motion correction
~~~~~~~~~~~~~~~~~

Motion correction realigns BOLD volumes to correct for subject motion. FSL
``mcflirt`` performs volume realignment (``func.motion_correction.dof``,
default 6). The reference volume (``ref_vol``) is the middle volume
(default), a given timepoint, or the temporal mean (via ``fslmaths -Tmean``
or ``fslroi``). Outputs include the motion-corrected 4D BOLD series, motion
matrices, and motion parameters (TSV).

Runs with fewer than 15 volumes skip motion correction, and pass-through
outputs (including a temporal mean and zero-filled motion parameters) are
generated.


Despike
~~~~~~~

Despiking removes transient intensity spikes from the BOLD time series
with AFNI ``3dDespike`` and local editing (cutoff parameters ``c1``/``c2``,
default 5/10). It is optional and controlled by ``func.despike.enabled``
(off by default). The first N volumes can be ignored via
``func.despike.ignore_first_volumes``.


Within-session coregistration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple BOLD runs exist per session, within-session
coregistration (``func.coreg_runs_within_session``, on by default)
aligns runs to a common reference and produces a session-averaged
temporal mean. This improves the stability of the temporal mean used for
bias field correction, conform, and registration.

Each run's temporal mean is registered to the first run's temporal mean
with ANTs rigid registration; the transform is then applied to that run's
4D BOLD series.


.. _bias-correction:

Bias field correction of the temporal mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To improve downstream registration, low-frequency intensity inhomogeneity is
corrected in the reference temporal mean image only. Bias field correction
is not applied to the full 4D time series, preserving the original temporal
signal for subsequent analyses.

N4 bias field correction (ANTs ``N4BiasFieldCorrection``) is applied to the
temporal mean of the motion-corrected BOLD series (and to anatomical images
in the anatomical workflow). Before N4, intensities may be rescaled to a
non-zero mean of 100 (configurable), and any negative voxels are clamped to
zero with a warning (N4 uses log-domain math). Anatomical bias field
correction uses a brain mask from skull stripping so background zeros are
excluded from the histogram; functional bias field correction runs earlier
in the pipeline (before conform and skull stripping) and therefore has no
brain mask at this step, so background zeros may remain in the image.


Conform and skull stripping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functional temporal mean is conformed to the selected anatomical brain
(or to the output template when there is no anatomical), using the same
strategy as anatomical conform: skull stripping the mean, FSL FLIRT rigid
registration to the template, resampling with AFNI ``3dresample``, and
application of the transform (optionally composed with anatomical
transforms).

For skull stripping, a UNet-based functional (EPI) skull stripping model,
derived from NHP-BrainExtraction/DeepBet, is run on the temporal mean to
obtain a brain mask (the region the model is most confident in, if it marks
more than one). The mask is resampled with the same transforms as the 4D
BOLD series and published alongside it; the BOLD data itself is not masked.


Registration
~~~~~~~~~~~~

ANTs registration (rigid, affine, or SyN as configured) maps the mean
functional image to the preprocessed anatomical or directly to the
template. Composite transforms are applied to the 4D BOLD series and brain
mask with ``antsApplyTransforms`` (e.g. BSpline interpolation for BOLD).
The result is the preprocessed BOLD series and mask in anatomical or
template space.


Temporal SNR (tSNR)
~~~~~~~~~~~~~~~~~~~

Brainana computes a quick functional-quality map for QC: voxelwise temporal
SNR (``|mean| / SD`` over time) of the preprocessed 4D BOLD series, saved as
per-run and session-average maps (``*_stat-tsnr_boldmap``) with optional
projection to the surface. Runs that are not 4D or have fewer than 10
timepoints are skipped.


Confound regressors
~~~~~~~~~~~~~~~~~~~

Brainana estimates fMRIPrep-compatible nuisance regressors for optional
downstream denoising. These are regressors only — the BOLD image is
never scrubbed or modified.

The step is controlled by ``func.confounds.enabled`` (on by default) and
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
convert rotations to millimeters.

The inputs are the preprocessed 4D BOLD series (T1w space preferred), motion
parameters (when motion correction is enabled), and a brain mask (required
for DVARS and global signal). Tissue regressors additionally require a T1w
segmentation and its label lookup table.

The method is a dependency-light reimplementation (NumPy/pandas/nibabel)
of the standard Power/Jenkinson/nipype formulas — not nipype itself.
Output is compatible with ``nilearn.interfaces.fmriprep.load_confounds``.
CompCor and cosine regressors are not produced. The regressors are:

- 24-parameter motion: ``trans_x/y/z`` and ``rot_x/y/z`` with their
  derivatives and squared terms.
- ``framewise_displacement`` (Power et al. 2012) and ``rmsd``
  (Jenkinson 1999). Both convert rotations to millimeters on a sphere of
  radius ``func.confounds.fd_radius_mm``, which defaults to the macaque
  27 mm; set it to the species' own head radius for other primates. FD
  values — and therefore ``fd_outlier_threshold_mm`` — are only comparable
  between runs computed at the same radius, so the radius actually used is
  recorded in the JSON sidecar.
- ``dvars`` and ``std_dvars``.
- ``global_signal`` (with expansions); ``csf``, ``white_matter`` and
  ``csf_wm`` are added only when a T1w segmentation is available.
- Outlier indicators: ``non_steady_state_outlier##`` and
  ``motion_outlier##``.

Outputs are a BIDS ``*_desc-confounds_timeseries.tsv`` plus a JSON
sidecar, and an fMRIPrep-style confounds panel in the QC report. The
confounds and motion QC figures shade the outlier frames as vertical
bands — gray for ``non_steady_state_outlier##``, red for
``motion_outlier##``. See :doc:`outputs` for the full column list.


Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Domain
     - Step
     - Main tool / method
   * - Anatomical
     - Ingest normalization
     - nibabel header repair (4D collapse, missing or conflicting qform/sform)
   * -
     - Synthesis
     - ANTs rigid + average
   * -
     - Conform
     - FLIRT + UNet skull stripping + 3dresample
   * -
     - Skull stripping and segmentation
     - FastSurfer-style CNN (fine-tuned)
   * -
     - Bias field correction
     - ANTs N4BiasFieldCorrection
   * -
     - Registration
     - FireANTs SyN (default) or ANTs
   * - Surface
     - Surface reconstruction
     - FastSurfer-style workflow + FreeSurfer
   * - Functional
     - Slice timing
     - AFNI 3dTshift
   * -
     - Motion correction
     - FSL mcflirt
   * -
     - Despike
     - AFNI 3dDespike (optional)
   * -
     - Within-session coregistration
     - ANTs rigid
   * -
     - Bias field correction
     - ANTs N4BiasFieldCorrection
   * -
     - Conform and skull stripping
     - FLIRT + UNet skull stripping + 3dresample
   * -
     - Registration
     - FireANTs SyN (default) or ANTs
   * -
     - tSNR
     - nibabel (``|mean| / SD`` over time)
   * -
     - Confounds
     - Custom fMRIPrep-style regressors

For outputs and directory layout, see :doc:`outputs`.

.. seealso::

   - :doc:`synthesis_level` — how a subject's anatomy is combined across sessions
   - :doc:`anat_selection_for_func` — how the T1w reference for fMRI is selected
   - :doc:`spaces_and_transforms` — spaces and transforms
