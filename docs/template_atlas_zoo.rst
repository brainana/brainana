:og:description: Non-human primate (NHP) templates and atlases supported by Brainana for macaque MRI registration and parcellation, including NMT2, used as the output_space.

.. meta::
   :description: Non-human primate (NHP) templates and atlases supported by Brainana for macaque MRI registration and parcellation, including NMT2, used as the output_space.
   :keywords: NHP templates, macaque atlas, NMT2, parcellation, registration, output space

Template and atlas zoo
======================

Brainana supports multiple NHP templates and atlases for
registration and parcellation. The overview below summarizes the
template and atlas options available in the pipeline.

.. figure:: _static/pipeline_details/template_atlas_zoo.png
   :alt: Overview of template and atlas zoo options.
   :align: center
   :width: 100%

   Templates and atlases bundled with Brainana.

Template zoo
------------

Download: `template_zoo/template <https://github.com/brainana/brainana/tree/main/template_zoo/template>`_

The following templates can be used as ``output_space`` (e.g. in
:ref:`command-line-arguments` or in the configuration). Choose a
template and/or resolution (e.g. ``NMT2Sym:res-05``) via the
:doc:`configuration generator <config_generator>` or a config YAML.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Template
     - Description
     - Resolutions
   * - NMT2Sym (`ref <https://doi.org/10.1016/j.neuroimage.2021.117997>`__)
     - NMT v2 symmetric template; ``NMT2Sym:res-05`` is the Brainana default
     - ``res-025``, ``res-05``, ``res-1``
   * - NMT2Asym (`ref <https://doi.org/10.1016/j.neuroimage.2021.117997>`__)
     - NMT v2 asymmetric template (left/right preserved)
     - ``res-05``
   * - MEBRAINS (`ref <https://doi.org/10.1162/imag_a_00077>`__)
     -
     - ``res-04``, ``res-05``
   * - Yerkes19 (`ref <https://doi.org/10.1523/JNEUROSCI.0493-16.2016>`__)
     -
     - ``res-05``
   * - D99 (`ref <https://doi.org/10.1016/j.neuroimage.2008.10.058>`__)
     -
     - ``res-025``, ``res-05``

You can also give a path to your own ``.nii``/``.nii.gz`` template; outputs then use the
space label ``template`` and no atlases are backprojected (see :ref:`custom-template`).

Atlas zoo
---------

Download: `template_zoo/atlas <https://github.com/brainana/brainana/tree/main/template_zoo/atlas>`_

ARM1–ARM6 (`ref <https://doi.org/10.1016/j.neuroimage.2021.117997>`__)
   Combined hierarchical macaque brain atlas.

   * ARM merges `CHARM <https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/atlas_charm.html>`_ for cortical regions
     and `SARM <https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/atlas_sarm.html>`_ for subcortical regions.
   * Six levels of parcellation granularity are available (1 = coarsest, 6 = finest).
   * Brainana performs individual ARM2 parcellations for T1w data.

Retinotopy (`ref <https://doi.org/10.1523/JNEUROSCI.0569-17.2017>`__)
   Group-average polar angle and eccentricity maps for mapping visual field
   representations (e.g. V1, V2, V3).

Somatotopy, MacBNA, D99, CortHierarchy, FuncNetwork, FuncConnGrad
   CortHierarchy is a cortical hierarchy, FuncNetwork gives resting-state networks, and
   FuncConnGrad gives functional connectivity gradients. See the download directory above
   for references.

Every atlas above is provided in all five template spaces (NMT2Sym, NMT2Asym,
MEBRAINS, Yerkes19, D99) and is backprojected for whichever one you choose.
FuncNetwork, CortHierarchy and FuncConnGrad cover the left hemisphere only.

.. note::

   FreeSurfer- and GIfTI-format surfaces with all of these atlases for NMT2Sym, NMT2Asym, MEBRAINS, Yerkes19 and D99
   are available at `macaque_template_surfaces <https://github.com/xingyu-liu/macaque_template_surfaces>`_.
