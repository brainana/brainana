:og:description: Try Brainana on the bundled example macaque MRI dataset — demo run command, expected run time, and expected output.

.. meta::
   :description: Try Brainana on the bundled example macaque MRI dataset — demo run command, expected run time, and expected output.
   :keywords: Brainana demo, example dataset, try Brainana, macaque MRI, BIDS, expected output, run time

.. _try-a-demo:

Try a demo
==========

Brainana ships with a small, ready-to-run BIDS dataset — one macaque subject with
two T1w runs, one T2w, and two resting-state fMRI runs — so you can exercise the
full pipeline (including multi-run anatomical selection) end to end.

Download just the demo dataset from the project repository:

.. code-block:: bash

   git clone --depth 1 --filter=blob:none --sparse https://github.com/brainana/brainana.git
   cd brainana
   git sparse-checkout set examples/dataset_example

This needs git ≥ 2.25; on older git, clone the whole repository instead with
``git clone https://github.com/brainana/brainana.git``.

Run it
------

From the repository root, point Brainana at ``examples/dataset_example`` as the input:

.. code-block:: bash

   docker run -it --rm --gpus all \
       -v "$(pwd)/examples/dataset_example":/input \
       -v <path/to/output_dir>:/output \
       -v <path/to/work_dir>:/output_wd \
       -v <path/to/license.txt>:/fs_license.txt \
       liuxingyu987/brainana:<version> /input /output \
       --work_dir /output_wd --freesurfer_license /fs_license.txt

Replace ``<version>`` with a published tag (e.g. ``2.1.0``), and create the output and
work directories beforehand so results are owned by you, not root (see
:ref:`docker-run-as-user-not-root`). Running without a GPU or without a FreeSurfer
license is covered in the :ref:`Docker user guide <usage-docker-guide>`.

Expected result
---------------

The full run finishes in about **20 minutes** with the GPU on a typical workstation
(8 CPU cores, 20 GB RAM, one NVIDIA GPU); CPU-only is roughly 60% slower.
Outputs follow the BIDS-derivatives layout in :doc:`outputs`, with a browsable HTML QC
report (``sub-example.html``) at the output root — see a
`sample report <_static/QCreport_example/sub-example.html>`_.
