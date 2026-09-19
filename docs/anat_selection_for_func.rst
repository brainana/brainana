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

.. mermaid::

   %%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eef2ff', 'primaryBorderColor': '#6366f1', 'primaryTextColor': '#1e1b4b', 'lineColor': '#6b7280', 'edgeLabelBackground': '#f8fafc'}, 'flowchart': {'curve': 'basis', 'htmlLabels': true, 'padding': 4, 'diagramPadding': 2, 'useMaxWidth': false}}}%%
   flowchart LR
       A{"Subject<br/>has T1w?"}
       A -->|No| B(["Template T1w"])
       A -->|Yes| SYNTH["<b>SYNTH</b>: multiple T1w<br/>runs per ses<br/>→ one T1w"]
       SYNTH --> D{"T1w in how<br/>many sess?"}
       D -->|one ses only| E(["That ses's T1w"])
       D -->|multiple ses| H{"anat.<br/>synthesis_level?"}
       H -->|"subject (default)"| I(["Subject-level T1w<br/><i>shared by all func ses</i>"])
       H -->|"session<br/>session_longitudinal"| J(["Same-ses T1w<br/><i>or lexicographically first<br/>other ses, if absent</i>"])

       classDef result fill:#ecfdf5,stroke:#059669,color:#064e3b
       classDef proc   fill:#eff6ff,stroke:#3b82f6,color:#1e3a8a
       classDef decision fill:#eef2ff,stroke:#6366f1,color:#1e1b4b
       class A,D,H decision
       class B,E,I,J result
       class SYNTH proc

.. .. code-block:: text

..    Subject has T1w?
..    │
..    ├─ No ──────────────────────────────────────────────► Template T1w
..    │
..    └─ Yes
..       │
..       │  [auto] multiple T1w runs in a session → synthesized to one T1w
..       │
..       T1w present in how many sessions?
..       │
..       ├─ One session only
..       │   ├─ Func session  = that session ─────────────► Same-session T1w
..       │   └─ Func session != that session ─────────────► That session's T1w
..       │
..       └─ Multiple sessions
..              │
..              anat.synthesis_level?
..              │
..              ├─ "subject"  (default) ─────────────────► Subject-level T1w
..              │                                           (synthesized across sessions;
..              │                                            shared by all func sessions)
..              └─ "session" ───────────────────────────► Same-session T1w
..                                                         (or lexicographically first
..                                                          other session, if absent)


Scenario 1 — No T1w for the subject
-----------------------------------

When no T1w image exists for the subject, all functional sessions
automatically use the specified **template T1w**.


Scenario 2 — T1w available for the subject
------------------------------------------

.. note::

   **Within-session synthesis** — when a session contains more than one T1w
   run, they are always synthesized into a single T1w before anything else.
   This step is automatic and applies to all cases below.

.. _anat-sel-case21:

Scenario 2.1 — T1w in only one session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Functional runs **in the same session** as the T1w use it directly.
- Functional runs **in any other session** also use that session's T1w as the anatomical reference.

.. _anat-sel-case22:

Scenario 2.2 — T1w in multiple sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When T1w images exist across multiple sessions, the
``anat.synthesis_level`` config option controls what happens:

.. code-block:: yaml

   anat:
     synthesis_level: "subject"   # default — or "session", "session_longitudinal"


- **``synthesis_level: "subject"`` (default)**

  All T1w images across sessions are combined into a single
  **subject-level** T1w, and every functional session of the subject uses that
  one shared T1w.

- **``synthesis_level: "session"``**

  Each session retains its own T1w. Functional runs in a session that has a T1w
  use it directly. Functional runs in a session without a T1w fall back to the
  T1w from the lexicographically first other session of the same subject.

- **``synthesis_level: "session_longitudinal"``**

  Anatomical selection is **identical to** ``"session"`` — every row in the
  chart above resolves the same way, and the functional stream is unaffected.
  What it adds happens later, inside surface reconstruction.

For what each level produces, and when to choose which, see
:doc:`synthesis_level`.
