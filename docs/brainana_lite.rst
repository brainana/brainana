:og:description: Brainana Lite is a notebook-based (Jupyter/Google Colab) volumetric T1w preprocessing workflow for a single macaque subject — no Docker or Nextflow required.

.. meta::
   :description: Brainana Lite is a notebook-based (Jupyter/Google Colab) volumetric T1w preprocessing workflow for a single macaque subject — no Docker or Nextflow required.
   :keywords: Brainana Lite, macaque T1w preprocessing, Colab neuroimaging, Jupyter, single subject

Brainana Lite
=============

Brainana Lite is a **notebook-based** workflow for volumetric **T1w** preprocessing of a single macaque subject, in **Jupyter** or **Google Colab**—no Docker or Nextflow. For production batch work, multiple modalities, surfaces, and the full HTML QC suite, use the full Docker pipeline (:doc:`installation`, :doc:`usage_notes`).

Run it
------

Open the notebook, edit ``WORKING_DIR`` in the **USER SETTINGS** cell, then **Run All**. Colab vs local Jupyter is detected automatically.

- **GitHub:** `BrainanaLite.ipynb <https://github.com/brainana/brainana/blob/main/examples/BrainanaLite.ipynb>`_
- **Colab:** Open in `Colab <https://colab.research.google.com/github/brainana/brainana/blob/main/examples/BrainanaLite.ipynb>`_

When to use it
--------------

- **Single subject, T1w only** — one macaque, anatomical T1w
- **No BIDS dataset required** — you only need NIfTI file(s) in a folder
- **No local system setup** — on Colab, a browser is enough
- **Quick, interactive runs** — trying a scan, teaching, or a one-off preprocess

What you get
------------

After **Run All**, derivatives are written in a **BIDS-styled** layout (exact filenames and the output tree are printed when preprocessing finishes):

- **Preprocessed T1w** in **individual (T1w) space** and in your chosen **template space** (skull-stripped brain volumes included)
- **Brain mask**, **hemisphere mask**, and **atlas-based tissue segmentation** in individual space (e.g. ARM2), with a color lookup table
- **Standard macaque atlases** backprojected into **individual spaces** (T1w and scanner)
- **An uncropped conformed T1w** (``desc-conformFullFOV``) — the conform field of view is sized from the template, so a recording chamber, head-post or the neck can fall outside it and be cropped from every other derivative; this leaf output keeps them visible
- **QC snapshot figures** (conformation — including a full-field-of-view figure with the processing box drawn on it — skull stripping, bias correction, segmentation, registration, and related checks)
- **A** ``dataset_description.json`` **at the derivatives root**, and the effective configuration of the run under ``lite_reports/``

Lite outputs are related to but not identical to the full pipeline; see :doc:`outputs` for the canonical Docker derivative layout. QC figures are displayed inline in the notebook as they are produced; the aggregated per-subject HTML QC report is a full-pipeline feature.

See also
--------

- :doc:`installation`
- :doc:`processing`
- :doc:`outputs`
- :doc:`faq`
