:og:description: Brainana FAQ and troubleshooting: config files, FreeSurfer license, running without a GPU, network drives, non-root use, Windows paths, and out-of-memory errors.

.. meta::
   :description: Brainana FAQ and troubleshooting: config files, FreeSurfer license, running without a GPU, network drives, non-root use, Windows paths, and out-of-memory errors.
   :keywords: Brainana FAQ, troubleshooting, FreeSurfer license, no GPU, network drive, out of memory

FAQ and troubleshooting
=======================

**Setup and configuration**

- `Do I need a config file?`_
- `Can I use my own template?`_
- `Can I run without a FreeSurfer license?`_
- `What if I don't have a compatible GPU?`_

**Running on your system**

- `Can I use a network drive for input or output?`_
- `I don't want Brainana to run as root. I want it to run as my own user. What should I do?`_
- `I'm on Windows — how do I write paths?`_

**Resources and troubleshooting**

- `How do I align container resources with Nextflow?`_
- `My pipeline run is hanging.`_
- `The pipeline fails with an out-of-memory error when I run it in Docker. What should I do?`_

----

Setup and configuration
------------------------

.. rst-class:: faq-question

Do I need a config file?
~~~~~~~~~~~~~~~~~~~~~~~~

No. Built-in defaults are used for all pipeline steps. To customise the pipeline you have two options:

1. **Config file (recommended):** Generate a YAML config file with the :ref:`generating-config-file` interactive generator (in :doc:`usage_notes`), mount it into the container (e.g. ``-v <path/to/config.yaml>:/config.yaml``), and pass ``--config /config.yaml``.
2. **Command-line arguments:** Pass common options directly in the ``docker run`` command (e.g. ``--anat_only``, ``--output_space "NMT2Sym:res-1"``). See :ref:`command-line-arguments`.

.. _custom-template:

.. rst-class:: faq-question

Can I use my own template?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Instead of a bundled template name, pass a path to your own ``.nii``/``.nii.gz``
template image as ``--output_space``. Mount the file into the container and pass its
in-container path:

.. code-block:: bash

   docker run --rm \
       -v <path/to/bids>:/input -v <path/to/out>:/output \
       -v <path/to/my_template.nii.gz>:/template.nii.gz \
       liuxingyu987/brainana:<version> /input /output --output_space /template.nii.gz

Outputs then use the BIDS space label ``template`` (e.g. ``*_space-template_*``). The
file must exist and end in ``.nii``/``.nii.gz`` or the run aborts at the start. A
custom template has no bundled atlases, so atlas outputs are skipped for that space.

.. rst-class:: faq-question

Can I run without a FreeSurfer license?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anatomical and functional preprocessing will still run, but surface reconstruction will be skipped. The container will warn if the license is missing.

Get a free license at https://surfer.nmr.mgh.harvard.edu/registration.html, then mount it with ``-v <path/to/license.txt>:/fs_license.txt`` and pass ``--freesurfer_license /fs_license.txt``.

.. rst-class:: faq-question

What if I don't have a compatible GPU?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run the pipeline without a GPU; it will use the CPU. Omit ``--gpus`` from your ``docker run`` command—no other options are needed.

If you do have an NVIDIA GPU and want to use it, add ``--gpus all`` and ensure the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ is installed on the host. See :doc:`installation` for setup steps.

.. _cpu-mode-with-gpu-host:

**To run on the CPU on a machine that has an NVIDIA GPU**, omit ``--gpus all`` rather than
passing it and setting ``general.gpu_device: -1`` in the config file. Without ``--gpus all``
the GPU driver is never exposed to the container, which is the most reliable way to run on
the CPU.

.. warning::

   **Windows (Docker Desktop with WSL2):** do not combine ``--gpus all`` with CPU mode.
   With the GPU exposed through WSL2 but not used, CPU-mode skull stripping has been seen to
   abort with exit status 134 and ``free(): double free detected`` in the task's
   ``.command.err``. The same run passes with ``--gpus all`` omitted. Either run on the GPU
   (``--gpus all`` with the default ``general.gpu_device: auto``) or leave ``--gpus all`` out.

.. rst-class:: faq-question

Which values does ``general.gpu_device`` take?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``auto`` (default): use the NVIDIA GPUs the container can see, otherwise the CPU.
- ``-1`` or ``cpu``: run every step on the CPU, even when a GPU is visible.
- ``0``, ``1``, …: run every GPU step on that one GPU (numbered as ``nvidia-smi`` lists them).
  The run stops at start-up if that GPU does not exist.

Steps that use the GPU are skull stripping and segmentation, the functional brain mask, and
FireANTs SyN registration. Every other step runs on the CPU and never sees a GPU. Apple
silicon GPUs (MPS) are not used.

----

Running on your system
----------------------

.. rst-class:: faq-question

Can I use a network drive for input or output?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**We recommend keeping the output directory and work directory on local storage.** Writing to a network drive (NFS, SMB, etc.) can cause permission errors, copy failures in early stages, or poor I/O performance and timeouts. If you see failures that look like permission or copy issues soon after the run starts, try pointing the output and work-directory mounts to local paths.

**Input on a network drive is fine.** You can leave your BIDS dataset on a network share and set the output (and work directory) to a local path. For example: mount the network BIDS root with ``-v <path/on/network/bids_dir>:/input`` and use local paths for ``-v <path/on/local/output_dir>:/output`` and ``-v <path/on/local/work_dir>:/output_wd``. The pipeline reads from the network and writes only to local disk.

.. _docker-run-as-user-not-root:

.. rst-class:: faq-question

I don't want Brainana to run as root. I want it to run as my own user. What should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before ``docker run``, ensure the **output** directory on your host already exists and is owned by your user. If you pass ``--work_dir`` (with a matching ``-v`` mount for the work directory), do the same for that host path. Mount those directories into the container with ``-v``.

That way Docker does not create those directories as root inside the container. 

Whenever practical, create fresh output and work directories (owned by you) instead of reusing old paths—this also avoids permission problems carried over from earlier runs.

.. _windows-paths:

.. rst-class:: faq-question

I'm on Windows — how do I write paths?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Docker volume mounts (``-v``) require the **host** path (the part before ``:``) to use
**forward slashes**, regardless of which Windows terminal you use.
The paths **inside** the container (after ``:``) are always Linux paths and never change.

**PowerShell or Command Prompt**

Replace backslashes with forward slashes in the host path:

.. code-block:: text

   # Replace this:
   #   -v C:\Users\me\bids:/input
   # With this:
   #   -v C:/Users/me/bids:/input

Full example:

.. code-block:: powershell

   docker run -it --rm `
       -v C:/Users/me/bids:/input `
       -v C:/Users/me/output:/output `
       -v C:/Users/me/work:/output_wd `
       liuxingyu987/brainana:<version> /input /output `
       --work_dir /output_wd

.. note::

   In PowerShell, use the backtick for line continuation instead of the backslash (``\``).

**WSL2 (Windows Subsystem for Linux)**

Use the ``/mnt/c/`` prefix to reference Windows drives:

.. code-block:: bash

   docker run -it --rm \
       -v /mnt/c/Users/me/bids:/input \
       -v /mnt/c/Users/me/output:/output \
       -v /mnt/c/Users/me/work:/output_wd \
       liuxingyu987/brainana:<version> /input /output \
       --work_dir /output_wd

----

Resources and troubleshooting
-----------------------------

.. rst-class:: faq-question

How do I align container resources with Nextflow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The container defaults to 8 CPUs and 20 GB for Nextflow (controlled by ``NXF_MAX_CPUS`` and ``NXF_MAX_MEMORY``). To change these:

- Pass ``-e NXF_MAX_CPUS=<n>`` and ``-e NXF_MAX_MEMORY=<n>g`` to ``docker run``.
- Use ``-profile minimal`` (4 CPUs, 16 GB) or ``-profile recommended`` (8+ CPUs, 32 GB) for preset profiles.

If either value is more than the container actually has (Docker Desktop's VM size on macOS and
Windows, or ``docker run --cpus/--memory``), the container lowers it to what is available (about
90% of the memory) and prints a ``WARNING`` at start-up. Steps that ask for more CPUs or memory
than that limit are given the limit instead.

See :ref:`command-line-arguments` for the full resource options.

.. rst-class:: faq-question

My pipeline run is hanging.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This typically happens when Nextflow runs out of memory. Try one or more of the following:

- Increase the RAM available to Docker.
- Use ``-profile minimal`` to reduce resource usage.
- Set ``-e NXF_MAX_CPUS`` and ``-e NXF_MAX_MEMORY`` to match your available resources.
- Resume from the last checkpoint by re-running the same command (Nextflow resume is enabled by default, provided the work directory is mounted — see :ref:`usage-docker-guide`).

.. rst-class:: faq-question

The pipeline fails with an out-of-memory error when I run it in Docker. What should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**On Docker Desktop (macOS/Windows/Linux with Docker Desktop):**

Docker Desktop runs containers inside a Linux virtual machine that has a configurable **Memory** limit. 
By default, this limit is **50% of your host RAM** (see `Docker Desktop advanced settings <https://docs.docker.com/desktop/settings-and-maintenance/settings/#advanced>`_).
If this default (for example, 4 GB on an 8 GB machine) is too low, the pipeline can run out of memory even though the host itself still has free RAM.

To fix this, open Docker Desktop and go to ``Settings → Resources → Advanced``. Increase the **Memory** allocation (for example, to 6–7 GB on an 8 GB machine), apply the changes, and rerun the pipeline.

**On a Mac with Apple silicon (M1 and later):** the Brainana image is built for x86-64, so Docker
Desktop runs it under emulation. It runs on the CPU only and is noticeably slower than on an
x86-64 machine with the same number of cores.

**On native Docker on Linux (no Docker Desktop):**

There is no Docker Desktop VM, so there is no extra global memory cap. Out-of-memory errors here usually mean either:
the container truly exceeds available system RAM, or you (or your system) have set explicit memory limits (for example, ``--memory 4g`` or cgroup limits).

In that case, you may need to run on a machine with more RAM and/or relax or remove those explicit Docker/cgroup memory limits.