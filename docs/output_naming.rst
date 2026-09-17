:og:description: How brainana names its output files — the BIDS entity vocabulary, one slot per concern, and the deliberate deviations from the specification.

.. meta::
   :description: How brainana names its output files — the BIDS entity vocabulary, one slot per concern, and the deliberate deviations from the specification.
   :keywords: BIDS entities, derivative naming, filename convention, desc entity, space entity

.. _output-naming:

Output naming
=============

Every file brainana publishes is named the same way: the entities the raw file
carried, plus one entity per thing brainana has to say about the product, plus a
BIDS suffix. This page states which entity means what, so a name can be read
without knowing which pipeline step wrote it.

:doc:`outputs` lists the actual filenames. This page is the rule behind them.

The four roles
--------------

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - Role
     - Entities
     - Who writes them
   * - identity
     - ``sub`` ``ses`` ``task`` ``acq`` ``ce`` ``dir`` ``rec`` ``run`` ``echo``
       ``flip`` ``inv`` ``mt`` ``part`` ``recording``, plus any entity brainana
       does not recognise
     - **The raw data only.** These describe what was acquired. brainana copies
       them onto its outputs and never sets or changes one.
   * - frame
     - ``space`` ``from`` ``to`` ``mode`` ``res`` ``den`` ``hemi``
     - brainana. Which reference frame the data are in — see
       :doc:`spaces_and_transforms`.
   * - variant
     - ``desc`` ``split``
     - brainana. Which processing variant this is: ``desc-preproc``,
       ``desc-brain``, or the step that produced a QC figure
       (``desc-conform``, ``desc-func2target``, …). At most one per name.
   * - product
     - ``atlas`` ``stat``
     - brainana. What kind of product this is: a parcellation, a statistical map.

The rule this encodes is that each concern has exactly one named slot. Writing a
pipeline concern into an identity entity looks harmless and is not: the entity
already means something, so the two meanings become indistinguishable, and the
value the scanner recorded is gone.

.. rubric:: A worked example

A subject's longitudinal reconstruction needs to say "this is the base-seeded
one, not the session's own". It carries ``space-base``:

.. code-block:: text

   sub-01_ses-003_run-1_desc-surfReconTissueSeg_T1w.png              the session's own surfaces
   sub-01_ses-003_run-1_space-base_desc-surfReconTissueSeg_T1w.png   base-seeded, in base space
   sub-01_space-base_desc-surfReconTissueSeg_T1w.png                 the base template itself

``space`` is the honest slot because the difference *is* a difference of frame —
which is the same reason the functional and fsnative-atlas streams keep using the
cross-sectional surfaces (see :ref:`longitudinal-surface-stream`). Marking it with
``acq`` instead, as an earlier version did, meant a subject scanned with two
protocols in one session lost that distinction, and both timepoints resolved to
one filename.

Why a name must round-trip
--------------------------

A name is parsed back into entities by the QC report, the atlas discovery, the
sidecar writer and the Viewer. Parsing keeps the **last** value of a repeated
entity, so a name carrying two ``desc-`` tokens silently loses one — the file is
present, and every consumer sees it claiming less than it is.

So no name may carry an entity twice. ``brainana`` enforces this where names are
built rather than checking afterwards, and
``python scripts/check_output_naming.py <output_dir>`` re-checks a finished run.

Deliberate deviations
---------------------

Three things in the output tree are not what the specification would give, and
are kept on purpose because a consumer depends on each:

``<prefix>_space-T1w_desc-preproc_T1w_brain.nii.gz``
   A ``_brain`` token after the suffix. Not a BIDS suffix, so a strict parser
   cannot read it back. The Viewer's base-volume fallback names it.

``atlas-<name>_space-<space>_sub-<id>[_ses-<id>].nii.gz``
   Atlas names lead with ``atlas-`` and carry ``sub-`` last. Atlas discovery — in
   the Viewer and in brainana itself — matches on that leading token.

An entity brainana does not recognise
   Passed through verbatim, in its original position, rather than dropped. Two
   runs that differ only by such an entity must not collapse onto one filename.

Adding an output
----------------

Choose the slot from the table above, and reuse an existing value where one fits:
:doc:`outputs` lists the ``desc`` and ``space`` values already in use, and a new
value that duplicates an old meaning is harder to undo than to avoid.
