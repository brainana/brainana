<p align="center">
  <img src="docs/_static/brainana_logo_side.png" alt="Brainana logo" width="500">
</p>

# Brainana: an end-to-end preprocessing framework for macaque neuroimaging

[![Documentation](https://readthedocs.org/projects/brainana/badge/?version=stable)](https://brainana.readthedocs.io/en/stable/)
[![Docker](https://img.shields.io/badge/docker-liuxingyu987%2Fbrainana-brightgreen.svg?logo=docker)](https://hub.docker.com/r/liuxingyu987/brainana/tags/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL--v3-blue.svg)](LICENSE)

<p align="center">
  <img src="docs/_static/pipeline_details/brainana_unified_framework.png" alt="Brainana pipeline overview schematic" width="100%">
</p>

**Brainana** provides anatomical and functional (fMRI) processing, image registration, surface reconstruction, fMRIPrep-compatible confound regressors, and HTML QC reports for macaque neuroimaging data in a reproducible workflow (FSL, ANTs, AFNI, FreeSurfer, Nextflow).

> **Status:** Research software – feature-complete for main workflows; bugs and edge cases are still possible.

## Getting started

Brainana runs as a reproducible **[Docker](https://hub.docker.com/r/liuxingyu987/brainana/tags/)** image. Read the **[Docs](https://brainana.readthedocs.io/en/stable/)**, start with [Installation](https://brainana.readthedocs.io/en/stable/installation.html), then [Usage notes](https://brainana.readthedocs.io/en/stable/usage_notes.html). 

To get a feel for the pipeline, [try the demo](https://brainana.readthedocs.io/en/stable/demo.html) on the bundled [`examples/dataset_example/`](examples/dataset_example) dataset.

## Brainana Lite

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xingyu-liu/brainana/blob/main/examples/BrainanaLite.ipynb)

Lightweight volumetric T1w preprocessing for a single subject (no functional MRI, no surfaces). Run interactively in Jupyter or Google Colab—no Docker required.

- **Notebook:** [`examples/BrainanaLite.ipynb`](examples/BrainanaLite.ipynb) — on Colab, use **Run all**; the first pass may restart the runtime once, then run again.
- **Docs:** [Brainana Lite guide](https://brainana.readthedocs.io/en/stable/brainana_lite.html) — Jupyter / Colab T1w workflow

For the full multi-modality pipeline with surfaces and QC reports, use the installation above for docker.

## Brainana Viewer

<p align="center">
  <img src="docs/_static/brainana_viewer_big.png" alt="Brainana Viewer — macaque brain surfaces, atlas overlays, and functional maps" width="100%">
</p>

To visualize brainana's outputs interactively, use the companion **[Brainana Viewer](https://github.com/brainana/brainana-viewer)** — a cross-platform viewer.

## Citation

If you use **Brainana**, please cite:

> [Brainana: an end-to-end preprocessing framework for macaque neuroimaging](https://www.biorxiv.org/content/10.64898/2026.06.03.729972v1.abstract)

Please also cite the toolboxes **Brainana** uses (FSL, ANTs, AFNI, FreeSurfer, FastSurfer, FireANTs, and any macaque templates). Detailed references can be found in the QC report.

## License

Copyright (c) the Brainana Developers. Licensed under the GNU Affero General Public License v3 (AGPL-3.0). See [LICENSE](LICENSE) for the full text. Some dependencies (e.g. FreeSurfer) have their own licenses; you must comply with those as well.
