:og:description: Run the Brainana macaque MRI preprocessing pipeline on a BIDS dataset: quick-start docker run command, output directory, and FreeSurfer license setup.

.. meta::
   :description: Run the Brainana macaque MRI preprocessing pipeline on a BIDS dataset: quick-start docker run command, output directory, and FreeSurfer license setup.
   :keywords: run Brainana, BIDS dataset, docker run, macaque MRI pipeline, FreeSurfer license

Usage notes
===========

The Brainana workflow takes a BIDS-formatted dataset as input and writes preprocessed outputs to a directory you specify.

Quick start
-----------

1. Prepare a valid BIDS dataset (see :ref:`the-bids-format`) and an output directory.
2. For surface reconstruction, prepare a FreeSurfer license (see :ref:`the-freesurfer-license-optional`).
3. Ensure the Brainana image is pulled as described in :doc:`installation`, then run:

   .. code-block:: bash

      docker run -it --rm --gpus all \
          -v <path/to/bids_dir>:/input \
          -v <path/to/output_dir>:/output \
          -v <path/to/work_dir>:/output_wd \
          -v <path/to/license.txt>:/fs_license.txt \
          liuxingyu987/brainana:<version> /input /output \
          --work_dir /output_wd --freesurfer_license /fs_license.txt

.. note::

   - **Replace ``<version>``** with a published Brainana tag from Docker Hub, for example ``3.0.0``. See the `Brainana image tags on Docker Hub <https://hub.docker.com/r/liuxingyu987/brainana/tags>`_ for the list of available versions.
   - **No compatible GPU?** Omit ``--gpus all``; the pipeline runs on CPU with no other changes. Details in :ref:`Check GPU access <installation-check-gpu-access>`.
   - **``<path/to/work_dir>``** is a host path for Nextflow's intermediate files. Without this mount, resume is impossible.
   - **Run as your user, not root:** Pre-create the output and work directories on the host and own them before mounting; see :ref:`docker-run-as-user-not-root`.
   - **Windows users:** See :ref:`windows-paths`.

No configuration file is required; built-in defaults are used. To customise the pipeline, see the Configuration file section below. For all options after the image name, see :ref:`command-line-arguments`.

.. _the-bids-format:

The BIDS format
---------------

The input dataset must be in valid `BIDS <https://bids.neuroimaging.io/getting_started/folders_and_files/folders.html>`_ format. We recommend validating your dataset with the free online `BIDS Validator <https://bids-standard.github.io/bids-validator/>`_.

Minimal example layout (dataset root with one subject, one session, anat + func)::

   ./   # dataset root
   ├── sub-banana
   │   └── ses-1
   │       ├── anat
   │       │   ├── sub-banana_ses-1_run-1_T1w.nii.gz
   │       │   └── sub-banana_ses-1_run-1_T1w.json   # optional
   │       └── func
   │           ├── sub-banana_ses-1_task-eat_run-1_bold.nii.gz
   │           └── sub-banana_ses-1_task-eat_run-1_bold.json   # optional
   └── <other_subjects>

If you start with DICOM, you can either:

(1) Use `dcm2niix <https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#General_Usage>`_ to convert DICOM to NIfTI and then manually reorganise and rename files to BIDS. Use ``-ba y`` so dcm2niix writes BIDS-compatible JSON sidecar files; you still need to create the BIDS folder structure and naming yourself.

(2) Use `bids converter <https://bids.neuroimaging.io/tools/converters.html>`_, which converts DICOM to NIfTI and organises output into BIDS for you.

If you only have a few images (e.g. one), option (1) is usually simpler; for a large dataset, option (2) is often better but needs a bit of extra setup.

.. tip::

   Want to try Brainana on a ready-made dataset first? See :doc:`demo`.

.. _the-freesurfer-license-optional:

The FreeSurfer license (optional)
----------------------------------

Brainana uses FreeSurfer for surface reconstruction, which requires a valid license.

Without a valid license, surface reconstruction fails. However, anatomical and functional preprocessing still run. The container warns if the license is missing.

**Get or locate the license**

- No license yet: obtain a free one at https://surfer.nmr.mgh.harvard.edu/registration.html
- FreeSurfer already configured on your machine: the license is usually at ``$FREESURFER_HOME/license.txt`` (check with ``ls $FREESURFER_HOME/license.txt``).

.. _generating-config-file:

Configuration file (optional)
------------------------------

Brainana uses built-in defaults, so you can run the standard pipeline without a configuration file. 

Use a YAML configuration file when you need to customise the pipeline (e.g. template space, registration type, or BIDS filtering).

**Creating a configuration file**

- Use the interactive `configuration generator <_static/config_generator.html>`_ to choose options and download a ready-to-use YAML file.

.. _usage-docker-guide:

Docker user guide
-----------------

This section covers running Brainana in Docker: volume mounts, example commands, and the full list of command-line options.

Mounts
~~~~~~

.. note::

   **Windows users:** replace ``<path/to/...>`` with a forward-slash host path, e.g.
   ``C:/Users/me/bids``. See :ref:`windows-paths` in the FAQ for full examples covering
   PowerShell, CMD, and WSL2.

**Mandatory mounts**

- Input (BIDS-formatted): ``-v <path/to/bids_dir>:/input``
- Output: ``-v <path/to/output_dir>:/output``

**Optional mounts**

- Work directory: ``-v <path/to/work_dir>:/output_wd`` — stores Nextflow's intermediate files and cache. **Required for resume to work.** 
- FreeSurfer license: ``-v <path/to/license.txt>:/fs_license.txt`` — mount the file prepared in :ref:`the-freesurfer-license-optional`; omit to run without surface reconstruction.
- Configuration file: ``-v <path/to/config.yaml>:/config.yaml`` — mount the file prepared in :ref:`generating-config-file`; omit to use built-in defaults.

Example commands
~~~~~~~~~~~~~~~~~

**With default config (surface reconstruction enabled)**

.. code-block:: bash

   docker run -it --rm --gpus all \
       -v <path/to/bids_dir>:/input \
       -v <path/to/output_dir>:/output \
       -v <path/to/work_dir>:/output_wd \
       -v <path/to/license.txt>:/fs_license.txt \
       liuxingyu987/brainana:<version> /input /output \
       --work_dir /output_wd --freesurfer_license /fs_license.txt

**With default config (surface reconstruction disabled)**

.. code-block:: bash

   docker run -it --rm --gpus all \
       -v <path/to/bids_dir>:/input \
       -v <path/to/output_dir>:/output \
       -v <path/to/work_dir>:/output_wd \
       liuxingyu987/brainana:<version> /input /output \
       --work_dir /output_wd

**With a custom config**

.. code-block:: bash

   docker run -it --rm --gpus all \
       -v <path/to/bids_dir>:/input \
       -v <path/to/output_dir>:/output \
       -v <path/to/work_dir>:/output_wd \
       -v <path/to/license.txt>:/fs_license.txt \
       -v <path/to/config.yaml>:/config.yaml \
       liuxingyu987/brainana:<version> /input /output \
       --work_dir /output_wd --freesurfer_license /fs_license.txt \
       --config /config.yaml


.. _command-line-arguments:

Command-line arguments
~~~~~~~~~~~~~~~~~~~~~~

The following options can be passed after the image name (or after ``bids_dir`` and ``output_dir``).

.. code-block:: text

   docker run ... liuxingyu987/brainana:<version> [bids_dir] [output_dir] \
       [--freesurfer_license PATH] [--config PATH | --config_file PATH] \
       [-w PATH | --work_dir PATH] [--no_resume] \
       [--subjects SUBJECT [SUBJECT ...]] [--sessions SESSION [SESSION ...]] \
       [--tasks TASK [TASK ...]] [--runs RUN [RUN ...]] \
       [--anat_only] [--output_space SPACE] [-profile PROFILE] [-h | --help]

**Positional arguments**

``bids_dir``
   BIDS root directory mounted into the container.

``output_dir``
   Output directory mounted into the container.

**General options**

``--freesurfer_license PATH``
   Path to the FreeSurfer license file *inside the container* (required when surface
   reconstruction is enabled). Use the same path as your volume mount,
   e.g. ``/fs_license.txt``.

   Default: (none)

``--config PATH``, ``--config_file PATH``
   Path to a custom YAML configuration file inside the container (optional; built-in
   defaults are used when omitted).

   Default: (built-in)

``-w PATH``, ``--work_dir PATH``
   Nextflow work directory (path inside the container).

   Default: ``/output_wd``. Mount a host directory here to persist the work
   directory across runs and enable resume, e.g. ``-v <path/to/work_dir>:/output_wd``.
   Without this mount the work directory is lost when the container exits.

``--no_resume``
   Disable Nextflow resume; restart the pipeline from scratch.

``-h``, ``--help``
   Print the full argument listing and exit (no processing).

**Options for filtering BIDS queries**

``--subjects SUBJECT [SUBJECT ...]``
   Restrict processing to the listed subject IDs (omit the ``sub-`` prefix).

   Default: (all subjects)

``--sessions SESSION [SESSION ...]``
   Restrict to specific session IDs (omit the ``ses-`` prefix).

   Default: (all sessions)

``--tasks TASK [TASK ...]``
   Restrict to specific task names (functional data only).

   Default: (all tasks)

``--runs RUN [RUN ...]``
   Restrict to specific run indices.

   Default: (all runs)

**Processing options**

``--anat_only``
   Run only the anatomical pipeline; skip functional processing.

   Default: ``False``

``--output_space SPACE``
   Template space for registered outputs. Either a bundled template spec in
   ``TEMPLATE_NAME[:DESCRIPTION]`` format — examples: ``NMT2Sym:res-1`` (1 mm),
   ``NMT2Sym:res-05`` (0.5 mm), ``T1w`` (native space) — **or** a path to a custom
   template image (``.nii`` / ``.nii.gz``). See the FAQ :ref:`custom-template` for details.

   Default: ``NMT2Sym:res-05``

**Resource options**

``-profile PROFILE``
   Nextflow resource profile. Choices:

   - ``minimal`` — 4 CPUs, 16 GB RAM
   - ``recommended`` — 8+ CPUs, 32 GB RAM

   Default: (built-in: 8 CPUs, 20 GB)

``NXF_MAX_CPUS`` (environment variable)
   Maximum number of CPUs for Nextflow (e.g. ``docker run -e NXF_MAX_CPUS=8 ...``).

``NXF_MAX_MEMORY`` (environment variable)
   Maximum memory for Nextflow (e.g. ``docker run -e NXF_MAX_MEMORY=20g ...``).
