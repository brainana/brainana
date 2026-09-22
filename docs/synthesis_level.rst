:og:description: How Brainana combines a subject's anatomical scans — the anat.synthesis_level setting, and the longitudinal surface stream that measures within-subject change over time.

.. meta::
   :description: How Brainana combines a subject's anatomical scans — the anat.synthesis_level setting, and the longitudinal surface stream that measures within-subject change over time.
   :keywords: anatomical synthesis, longitudinal MRI, macaque surface reconstruction, within-subject template, cortical thickness change

.. _synthesis-level:

Anatomical synthesis level
==========================

``anat.synthesis_level`` decides how a subject's anatomical scans are combined
before anything else runs, and therefore how many cortical surfaces you get and
whether they can be compared across sessions.

.. code-block:: yaml

   anat:
     synthesis_level: "subject"   # default — or "session", "session_longitudinal"


Choosing a level
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 26 26 28

   * - Level
     - How anatomy is combined
     - Surfaces produced
     - Choose it when
   * - ``subject`` *(default)*
     - One T1w synthesized across all of the subject's sessions.
     - One reconstruction per subject.
     - Sessions are equivalent and you want a single anatomy per animal.
   * - ``session``
     - Each session keeps its own T1w.
     - One independent reconstruction per session.
     - Sessions should not be pooled, and you do not need vertices to
       correspond between them.
   * - ``session_longitudinal``
     - Same as ``session``, plus an unbiased base template per subject.
     - The per-session reconstructions **and** base-seeded ones that share a
       single mesh.
     - You intend to measure within-subject change over time.

Which T1w a *functional* run is registered to follows from this setting; see
:doc:`anat_selection_for_func` for that chart and its fallback rules.
``subject`` and ``session`` need nothing further — the rest of this page is the
longitudinal stream.


.. _longitudinal-surface-stream:

The longitudinal stream
-----------------------

What it does
~~~~~~~~~~~~

``session_longitudinal`` adds a second surface pass on top of the per-session
reconstruction, so that every session of a subject ends up sharing **one
cortical mesh**. Vertex *i* is then the same anatomical point at every
timepoint, and thickness at that vertex can be differenced across sessions
directly — no surface registration, and no spherical registration step.

When to use it
~~~~~~~~~~~~~~

- Set ``anat.synthesis_level: "session_longitudinal"``, which also requires
  ``anat.surface_reconstruction.enabled``.
- A subject needs **at least two sessions carrying anatomy**. Single-session
  subjects are still processed cross-sectionally; they are simply skipped by
  this stream, since an "unbiased template" built from one scan is that scan.
- Anatomical selection is unchanged from ``"session"``, and the functional
  stream is unaffected.

How it works
~~~~~~~~~~~~

.. mermaid::

   %%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eef2ff', 'primaryBorderColor': '#6366f1', 'primaryTextColor': '#1e1b4b', 'lineColor': '#6b7280', 'edgeLabelBackground': '#f8fafc'}, 'flowchart': {'curve': 'basis', 'htmlLabels': true, 'padding': 4, 'diagramPadding': 2, 'useMaxWidth': false}}}%%
   flowchart LR
       S1["ses-01 T1w"]
       S2["ses-02 T1w"]
       S3["ses-03 T1w"]
       RT["<b>1. Base template</b><br/>mri_robust_template<br/><i>robust rigid + median average</i>"]
       BASE(["<b>sub-01_base</b><br/>segmented and fully<br/>reconstructed<br/><i>one mesh</i>"])
       TP["<b>2. Per timepoint</b><br/>resample into base space,<br/>seed from the base mesh<br/><i>placed on each session's<br/>own intensities</i>"]
       L1(["sub-01_ses-01_long"])
       L2(["sub-01_ses-02_long"])
       L3(["sub-01_ses-03_long"])
       ST(["<b>Change statistics</b><br/>per-vertex rate, mean,<br/>percent change + ROI table<br/><i>written into sub-01_base/</i>"])

       S1 --> RT
       S2 --> RT
       S3 --> RT
       RT --> BASE
       BASE --> TP
       TP --> L1
       TP --> L2
       TP --> L3
       L1 --> ST
       L2 --> ST
       L3 --> ST

       classDef result fill:#ecfdf5,stroke:#059669,color:#064e3b
       classDef proc   fill:#eff6ff,stroke:#3b82f6,color:#1e3a8a
       classDef input  fill:#f8fafc,stroke:#94a3b8,color:#334155
       class S1,S2,S3 input
       class RT,TP proc
       class BASE,L1,L2,L3,ST result

**1. Within-subject base template.** Each session's conformed volumes are
registered into a common unbiased space with ``mri_robust_template`` (robust
rigid registration, median averaging) and averaged. The average is then
segmented and fully reconstructed, giving one mesh for the subject.
``mri_robust_template`` is used on its own deliberately: it makes no atlas or
species assumptions, so it transfers to macaque data unchanged, where
FreeSurfer's ``recon-all -base``/``-long`` streams would run a human volume
pipeline and discard the CNN segmentation these surfaces are built on.

**2. Per-timepoint reconstruction.** Each session's volume is resampled into
base space and its reconstruction is seeded from the base's surfaces.
Tessellation through topology correction are inherited rather than recomputed,
while surface *placement* onward still runs against that session's own
intensities — so genuine change is still measured. Placement is anchored to the
base and capped, as ``recon-all -long`` does.

Time variable and change statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the meshes correspond vertex-for-vertex, a rate of change is fitted per
vertex and per ROI across the subject's timepoints.

The time value for each session is taken from
``<bids_dir>/sub-<id>/sub-<id>_sessions.tsv``, in this order:

1. the column named by ``anat.surface_reconstruction.longitudinal.time_column``,
   if you set one;
2. otherwise ``age``, then ``acq_time`` (converted to days from the subject's
   first session).

With no usable column, the fit falls back to any number in the session label,
and then to scan order — in which case the fitted rate is **per scan rather
than per unit time**. Which regime was used is recorded as ``time_source`` in
the run's change-statistics summary, because a rate is not interpretable
without it.

What you get
~~~~~~~~~~~~

- ``fastsurfer/sub-<id>_base/`` — the base template, reconstructed once, and
  the change statistics fitted across the subject's timepoints.
- ``fastsurfer/sub-<id>_ses-<id>_long/`` — one per session, in base space,
  sharing the base's vertex numbering.
- Base-space derivatives, atlases and QC figures, marked ``space-base``.

Longitudinal mode does not replace the per-session pipeline. Brainana still
writes the same **cross-sectional** FastSurfer trees as ``synthesis_level:
"session"`` — ``fastsurfer/sub-<id>_ses-<id>/``, each in that session's own T1w
space. The ``*_base`` and ``*_long`` directories and anything tagged
``space-base`` are **additional** outputs in the within-subject base reference.
:doc:`outputs` lists every path.

.. note::

   **Use longitudinal outputs for across-time surface analysis.** Shared vertex
   numbering on ``*_long`` meshes, plus the change statistics under ``*_base/``,
   is what lets you compare thickness, area, and ROI rates between sessions
   without registering surfaces to each other.

   **Use cross-sectional outputs for fMRI and fsnative atlases.** Functional
   preprocessing coregisters each run to the T1w chosen for that session (see
   :doc:`anat_selection_for_func`), then registers to template space and can
   project maps onto the cortical surface. Those steps always follow the
   cross-sectional ``fastsurfer/sub-<id>_ses-<id>/`` tree — not ``*_long`` or
   ``*_base``. The same cross-sectional surfaces drive ``space-fsnative``
   atlases under ``anat/atlas_space-fsnative/``.

   **Why the split:** In a ``*_long`` tree the session T1w has already been
   resampled into the base template before reconstruction, so ``orig.mgz`` and
   the mesh sit on the **base** voxel grid. Preprocessed BOLD still lives in
   **session-native** space, aligned to the cross-sectional T1w. Feeding
   functional volumes into the longitudinal mesh as if it were a normal session
   recon (for example ``mri_vol2surf`` with ``--regheader``) ignores the rigid
   timepoint-to-base transform in ``fastsurfer/sub-<id>_base/mri/transforms/``
   and misaligns data by exactly that amount.
