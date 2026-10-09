:og:description: Install Brainana for macaque MRI preprocessing via Docker (full anatomical + functional pipeline) or the Colab/Jupyter-based Brainana Lite T1w workflow.

.. meta::
   :description: Install Brainana for macaque MRI preprocessing via Docker (full anatomical + functional pipeline) or the Colab/Jupyter-based Brainana Lite T1w workflow.
   :keywords: install Brainana, macaque MRI Docker, NHP neuroimaging, Brainana Lite Colab

.. role:: bn-tag

Installation
============

Brainana can be installed in two ways:

.. container:: bn-choices

   .. container:: bn-choice bn-choice-primary

      .. rst-class:: bn-choice-title

      Docker :bn-tag:`Recommended`

      The full pipeline: anatomical and functional preprocessing, surface
      reconstruction, and HTML QC reports.

      .. rst-class:: bn-choice-meta

      Best for whole datasets and production runs.

      .. rst-class:: bn-choice-link

      :ref:`Set up Docker → <install-docker>`

   .. container:: bn-choice bn-choice-secondary

      .. rst-class:: bn-choice-title

      Brainana Lite

      A lightweight volumetric T1w workflow in Jupyter or Google Colab, with no
      Docker image.

      .. rst-class:: bn-choice-meta

      Best for trying Brainana on a single subject.

      .. rst-class:: bn-choice-link

      :doc:`Open Brainana Lite → <brainana_lite>`

.. _install-docker:

Docker
------

Docker is the recommended way to run the full Brainana pipeline.

System requirements
~~~~~~~~~~~~~~~~~~~

Brainana runs on any OS supported by Docker (Linux, macOS, or Windows with WSL2).

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Resource
     - Minimum
     - Recommended
     - Production
   * - RAM + swap
     - 16 GB
     - 20 GB+ for the full pipeline
     - 32 GB
   * - CPU
     - 4 logical cores
     - 8
     - 8+
   * - Disk
     - 20 GB
     - 50 GB+ for multiple subjects
     - 100 GB+
   * - GPU (optional, NVIDIA only)
     - 6 GB VRAM
     - 1 GPU, 6 GB+ VRAM
     - 1 GPU, 10 GB+ VRAM

A GPU also needs NVIDIA driver 525.60.13 or later and CUDA 12.0 or later. Without one,
everything runs on the CPU.

Set up Docker
~~~~~~~~~~~~~

Install Docker
^^^^^^^^^^^^^^

Install Docker if you do not have it (`Docker installation <https://docs.docker.com/get-docker/>`_),
then test it with the hello-world image:

.. code-block:: bash

   docker run -it --rm hello-world

You should see a message saying that Docker is working correctly.

.. _installation-check-gpu-access:

Check GPU access (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A GPU is compatible if it is NVIDIA and meets the driver and CUDA minimums. Check with:

.. code-block:: bash

   nvidia-smi

In the output (top-right corner), the driver version must be 525.60.13 or later and the
CUDA version 12.0 or later. If you have no NVIDIA GPU, or either value is below the
minimum, no compatible GPU is available; skip the rest of this step.

Otherwise, verify that Docker can access your GPU:

.. code-block:: bash

   docker run -it --rm --gpus all hello-world

If you see an error about ``nvidia-container-cli`` or ``libnvidia-ml.so``, make sure the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
and drivers are installed.

Pull the Brainana image
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   docker pull liuxingyu987/brainana:<version>

Replace ``<version>`` with a published Brainana tag, for example ``3.2.0`` (see the
`image tags on Docker Hub <https://hub.docker.com/r/liuxingyu987/brainana/tags>`_).
This is a one-time ~9 GB download (~23 GB on disk), typically a few minutes on a fast
connection and longer on slower networks.

Once the image is ready, see :doc:`usage_notes` to run the pipeline.
