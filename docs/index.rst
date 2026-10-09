:og:description: Brainana is an end-to-end macaque (non-human primate) MRI preprocessing pipeline: anatomical and functional (fMRI) preprocessing, registration, and surface reconstruction from BIDS data, reproducible with Docker and Nextflow.

.. meta::
   :description: Brainana is an end-to-end macaque (non-human primate) MRI preprocessing pipeline: anatomical and functional (fMRI) preprocessing, registration, and surface reconstruction from BIDS data, reproducible with Docker and Nextflow.
   :keywords: macaque MRI preprocessing, NHP neuroimaging, non-human primate fMRI, BIDS, surface reconstruction, ANTs, FreeSurfer, Nextflow

Brainana
========

About
-----

.. figure:: _static/pipeline_details/brainana_unified_framework.png
   :alt: Brainana unified framework for macaque MRI: anatomical, functional, and surface processing
   :align: center
   :width: 100%

   Anatomical, functional, and surface processing in one framework.

Brainana is a unified, end-to-end preprocessing framework for macaque (non-human primate) MRI. It provides anatomical and functional preprocessing, registration, tissue segmentation, and cortical surface reconstruction from BIDS data, reproducible with Docker and Nextflow and built on FSL, ANTs, AFNI, FreeSurfer, and FastSurfer.

.. figure:: _static/pipeline_details/pipeline_overview.png
   :alt: Brainana pipeline overview schematic
   :align: center
   :width: 100%

   Pipeline overview. A: image preprocessing (green: structural, purple: functional).
   B: dataset batch processing. C: benchmark runtime. D: Brainana Viewer.

To get started, see :doc:`installation` and :doc:`usage_notes`. Brainana runs with
built-in defaults; to change templates, registration, or BIDS filtering, build a
configuration file in your browser with the :doc:`configuration generator <config_generator>`.

To explore a subject's outputs interactively, use the companion
`Brainana Viewer <https://github.com/brainana/brainana-viewer>`_.

License
-------

Copyright (c) the Brainana Developers.
Licensed under the GNU Affero General Public License v3 (AGPL-3.0).

Citation
--------

Brainana: an end-to-end preprocessing framework for macaque neuroimaging `[preprint] <https://www.biorxiv.org/content/10.64898/2026.06.03.729972v1.abstract>`_

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 1
   :caption: User guide

   usage_notes
   config_generator
   demo

.. toctree::
   :maxdepth: 1
   :caption: Brainana Lite

   brainana_lite

.. toctree::
   :maxdepth: 1
   :caption: Processing and outputs

   processing
   outputs
   synthesis_level
   anat_selection_for_func
   spaces_and_transforms

.. toctree::
   :maxdepth: 1
   :caption: Templates and atlases

   template_atlas_zoo

.. toctree::
   :maxdepth: 1
   :caption: Viewer

   viewer

.. toctree::
   :maxdepth: 1
   :caption: Other info

   faq
