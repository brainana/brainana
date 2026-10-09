:og:description: How Brainana selects the T1w anatomical reference for macaque fMRI registration, based on available data and the anat.synthesis_level setting.

.. meta::
   :description: How Brainana selects the T1w anatomical reference for macaque fMRI registration, based on available data and the anat.synthesis_level setting.
   :keywords: anatomical reference, functional registration, macaque fMRI, T1w selection

T1w reference for fMRI
======================

When functional registration is enabled, the T1w reference is selected
based on what data are available for the subject and on the
``anat.synthesis_level`` configuration setting (see :ref:`anat-sel-case22`).


Overview
--------

The chart below shows how the reference is chosen; the sections after it
describe each case.

.. mermaid::
   :caption: T1w reference selection. Purple: one of the subject's own T1w images; green: the template T1w, used when the subject has none.

   %%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Lato, Helvetica Neue, Arial, sans-serif', 'fontSize': '13px', 'primaryColor': '#ffffff', 'primaryBorderColor': '#b4b9c1', 'primaryTextColor': '#333333', 'lineColor': '#777777', 'edgeLabelBackground': '#ffffff'}, 'flowchart': {'curve': 'basis', 'htmlLabels': true, 'padding': 10, 'diagramPadding': 2, 'nodeSpacing': 28, 'rankSpacing': 32, 'useMaxWidth': false}}}%%
   flowchart TB
       A{"Subject<br/>has a T1w?"}
       A -->|No| B(["Template T1w"])
       A -->|Yes| SYNTH["<b>Synthesis</b><br/>multiple T1w runs<br/>per session → one T1w"]
       SYNTH --> D{"T1w in how<br/>many sessions?"}
       D -->|one session| E(["That session's T1w"])
       D -->|multiple sessions| H{"anat.<br/>synthesis_level?"}
       H -->|"subject (default)"| I(["Subject-level T1w<br/><i>shared by all<br/>functional sessions</i>"])
       H -->|"session<br/>session_longitudinal"| J(["Same-session T1w<br/><i>or, if absent, the<br/>lexicographically first<br/>other session</i>"])

       classDef neutral fill:#ffffff,stroke:#b4b9c1,color:#333333,stroke-width:1.3px
       classDef purple  fill:#f1ecfb,stroke:#9370db,color:#333333,stroke-width:1.3px
       classDef green   fill:#e8f6ee,stroke:#3cb371,color:#333333,stroke-width:1.3px
       class A,D,H,SYNTH neutral
       class E,I,J purple
       class B green


No T1w for the subject
----------------------

When no T1w image exists for the subject, all functional sessions
automatically use the specified template T1w.


T1w available for the subject
-----------------------------

When a session contains more than one T1w run, they are always synthesized
into a single T1w before anything else. This within-session synthesis is
automatic and applies to both cases below.

.. _anat-sel-case21:

T1w in only one session
~~~~~~~~~~~~~~~~~~~~~~~

- Functional runs in the same session as the T1w use it directly.
- Functional runs in any other session also use that session's T1w as the anatomical reference.

.. _anat-sel-case22:

T1w in multiple sessions
~~~~~~~~~~~~~~~~~~~~~~~~

When T1w images exist across multiple sessions, the
``anat.synthesis_level`` config option controls what happens:

.. code-block:: yaml

   anat:
     synthesis_level: "subject"   # default — or "session", "session_longitudinal"


The levels select the reference as follows:

``synthesis_level: "subject"`` (default)
   All T1w images across sessions are combined into a single subject-level
   T1w, and every functional session of the subject uses that one shared T1w.

``synthesis_level: "session"``
   Each session retains its own T1w. Functional runs in a session that has a
   T1w use it directly. Functional runs in a session without a T1w fall back
   to the T1w from the lexicographically first other session of the same
   subject.

``synthesis_level: "session_longitudinal"``
   Anatomical selection is identical to ``"session"`` — every row in the
   chart above resolves the same way, and the functional stream is
   unaffected. What it adds happens later, inside surface reconstruction.

For what each level produces, and when to choose which, see
:doc:`synthesis_level`.
