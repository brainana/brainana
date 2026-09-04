:og:description: Install Brainana for macaque MRI preprocessing via Docker (full anatomical + functional pipeline) or the Colab/Jupyter-based Brainana Lite T1w workflow.

.. meta::
   :description: Install Brainana for macaque MRI preprocessing via Docker (full anatomical + functional pipeline) or the Colab/Jupyter-based Brainana Lite T1w workflow.
   :keywords: install Brainana, macaque MRI Docker, NHP neuroimaging, Brainana Lite Colab

Installation
============

Brainana can be installed in two ways:

- **Docker** (below) — full pipeline: anatomical and functional preprocessing, surface reconstruction, and HTML QC reports. This is the recommended path for most users.
- ✨ **Brainana Lite** — lightweight volumetric T1w workflow via Jupyter or Colab; no Docker image. See `BrainanaLite.ipynb <https://github.com/brainana/brainana/blob/main/examples/BrainanaLite.ipynb>`_ notebook on GitHub and the documentation page :doc:`brainana_lite`.

Docker
------

The recommended way to run the **full** Brainana pipeline is with Docker. 

System requirements
~~~~~~~~~~~~~~~~~~~

.. warning::

   - **Host:** Any OS supported by Docker (Linux, macOS, Windows with WSL2)
   - **RAM + swap:** ≥ 16 GB (recommended 20 GB+ for full pipeline)
   - **Disk:** ≥ 20 GB (recommended 50 GB+ for multiple subjects)
   - **CPU:** ≥ 4 logical cores (recommended 8+)
   - **GPU (optional, NVIDIA only):** ≥ 6 GB VRAM (recommended ≥ 10 GB for production)
   - **NVIDIA Driver (optional):** ≥ 520.61.05 if using GPU
   - **CUDA (optional):** ≥ 11.8 if using GPU

Resource guidelines:

- **Minimal:** 16 GB RAM, 4 CPUs, 20 GB disk
- **Recommended:** 20 GB RAM, 8 CPUs, 50 GB disk, 1 NVIDIA GPU with ≥ 6 GB VRAM
- **Production:** 32 GB RAM, 8+ CPUs, 100 GB+ disk, 1 NVIDIA GPU with ≥ 10 GB VRAM

Set up Docker
~~~~~~~~~~~~~

1. **Install Docker** if you do not have it (`Docker Installation <https://docs.docker.com/get-docker/>`_).

2. **Test Docker** with the hello-world image:

   .. code-block:: bash

      docker run -it --rm hello-world

   You should see a message indicating that Docker is working correctly.

.. _installation-check-gpu-access:

3. **Check GPU access** (optional):

   A GPU is compatible if it is NVIDIA and meets the driver and CUDA minimums. 
   
   1. You can check with:

   .. code-block:: bash

      nvidia-smi

   In the output (top-right corner), check:

   - Driver version — must be ≥ 520.61.05
   - CUDA version — must be ≥ 11.8

   If you have no NVIDIA GPU, or either value is below the minimum, no compatible GPU is available. Skip the rest of this step.

   2. verify Docker can access your GPU:

   .. code-block:: bash

      docker run -it --rm --gpus all hello-world

   .. note::

      If you see an error about ``nvidia-container-cli`` or ``libnvidia-ml.so``, ensure the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ and drivers are installed.

4. **Pull the Brainana image:**

   .. code-block:: bash

      docker pull liuxingyu987/brainana:<version>

   .. note::

      Replace ``<version>`` with a published Brainana tag from Docker Hub, for example ``2.1.0``. See the `Brainana image tags on Docker Hub <https://hub.docker.com/r/liuxingyu987/brainana/tags>`_ for the list of available versions.

   This is a one-time ~9 GB download (~23 GB on disk), typically a few minutes on a fast connection and longer on slower networks.

Once the image is ready, see :doc:`usage_notes` to run the pipeline.
