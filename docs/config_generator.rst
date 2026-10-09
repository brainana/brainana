:og:description: Build a Brainana configuration file in your browser: every pipeline option grouped by stage, with defaults filled in, ready to download as YAML.

.. meta::
   :description: Build a Brainana configuration file in your browser: every pipeline option grouped by stage, with defaults filled in, ready to download as YAML.
   :keywords: Brainana configuration, config YAML, macaque MRI pipeline options, configuration generator

.. _config-generator:

Configuration generator
=======================

Build a Brainana configuration file in your browser: pick options, read what each
one does, and download a ready-to-use YAML file. No configuration file is needed
to run the standard pipeline; use one when you want to change the template space,
registration, BIDS filtering, or any other default.

.. container:: bn-feature

   .. figure:: _static/pipeline_details/config_generator.png
      :alt: The configuration generator: a sidebar of option groups and a form of pipeline options with their defaults.
      :width: 100%
      :target: _static/config_generator.html

      The configuration generator. Click to open it.

   Every option, grouped by stage
      General, BIDS filtering, template, anatomical, functional, and registration
      settings, each with a short explanation.

   Defaults filled in
      Change only what you need. Advanced settings stay folded away until you open them.

   Copy or download
      Generate the YAML, then copy it to the clipboard or save it as a file.

   Covers every option
      A test checks that every option in the pipeline's defaults has a field in
      the generator.

   .. rst-class:: bn-button

   `Open the configuration generator → <_static/config_generator.html>`_

Use the file
------------

Mount the file into the container and pass it with ``--config``:

.. code-block:: text

   docker run ... -v <path/to/config.yaml>:/config.yaml \
       liuxingyu987/brainana:<version> /input /output ... --config /config.yaml

The file only needs the settings you change; anything left out keeps its default.
See the :ref:`full command <usage-custom-config>` in the usage notes.
