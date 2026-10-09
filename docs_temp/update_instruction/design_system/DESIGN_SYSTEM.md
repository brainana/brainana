# Brainana design system

The single source of truth for the look of every Brainana user-facing surface: the **QC
report**, the **configuration generator**, and the **BrainanaLite notebook** header.

Open **`styleguide.html`** (in this folder) in a browser for a live reference — color
swatches, the type scale, and every component. **`brainana-tokens.css`** is the canonical
token + component stylesheet that page links.

## How it's distributed

The tokens are defined once (below) and applied per surface in the way that surface allows:

| Surface | Where the tokens live | Why |
| --- | --- | --- |
| QC report | inlined in `src/nhp_mri_prep/quality_control/reports.py` as `_REPORT_CSS` (+ the dropdown JS `_REPORT_JS`) | reports are portable single HTML files — no sibling CSS |
| Config generator | a `<style id="bn-design-system">` block in `docs/_static/config_generator.html` | additive override over the page's original CSS |
| Notebook header | inline `style="…"` attributes in `examples/BrainanaLite.ipynb` cell 0 | Colab sanitizes `<style>`/external CSS in markdown cells |
| Style guide | links `brainana-tokens.css` directly | it lives next to the file |

The **RTD docs** (`docs/*.rst`) are a separate surface and do **not** use these tokens: they keep
`sphinx_rtd_theme` with a purple/green accent swap in `docs/_static/custom.css`. Their rules
(palette, diagrams, emphasis, admonitions, figures) live in **`styleguide_doc.html`** in this folder.

When you change a token, update `brainana-tokens.css` here **and** mirror the value into
the three surfaces above (they each carry their own copy by necessity).

## Tokens

| token | value | role |
| --- | --- | --- |
| `--bn-bg` | `#fffeee` | page background (cream) |
| `--bn-surface` | `#fffddd` | chips, code, nav, sidebars |
| `--bn-inset` | `#ffffff` | inset white panels / report body |
| `--bn-ink` | `#1b1b16` | headings |
| `--bn-text` | `#333333` | body text |
| `--bn-muted` | `#6b6651` | secondary text |
| `--bn-border` | `#e6e1cd` | airy hairlines |
| `--bn-border-mid` | `#cdc9b0` | chips / inputs / figure frames (more definition) |
| `--bn-code-bg` | `#f7f6ef` | inline code + code blocks |
| `--bn-accent` | `#fff27a` | signature yellow — heading underlines, brand chip, hover tints |
| `--bn-link` | `#8a7a00` (hover `#6f6300`) | links / interactive text (accessible mustard) |
| `--bn-ok` | `#3f6b46` on `#eef3ea` / `#cdddc2` | success / pass |
| `--bn-fail` | `#9c4636` on `#f8ece8` / `#e7c8bd` | failure / error |
| `--bn-warn` | `#8a6400` on `#fdf3d6` / `#e6d8a0` | warning |
| radius | card `10px` · inset `8px` · chip `6px` · pill `999px` | |
| font (sans) | `"IBM Plex Sans", system-ui, …` | all reading text |
| font (mono) | `"IBM Plex Mono", ui-monospace, …` | code / paths / values only |
| rhythm | `16px` / line-height `1.5`; paragraphs `16px`; headings `24/16`, line-height `1.25`; prose measure `760px` | GitHub-calibrated |

## Rules

- **Sans everywhere** (IBM Plex Sans). Monospace is reserved for literal code, file paths,
  and config values. Plex loads via a Google-Fonts `@import` with a system-sans fallback
  (offline reports render in the system font).
- **Yellow (`--bn-accent`) is decorative** — heading underlines, the brand chip, hover
  tints. Never used for body text and never to signal state.
- **State uses the semantic green/red/amber**, tuned earthy so they sit with the cream.
- **One interactive accent**: links/interactive text use `--bn-link` (mustard).
- **GitHub-style reading comfort**: the rhythm tokens above (16/1.5, generous heading and
  paragraph spacing, a ~760px prose measure, light hairlines, calm code blocks) are the
  source of the comfortable feel — not the typeface.

## QC report — surface-specific notes

- **No yellow in dividers** (a deliberate QC choice): section rules are gray
  (`--bn-border-mid`). The brand colour appears only as the `brainana` chip in the top bar.
- **Hierarchy** (simplest-but-clear): section (H1, gray rule) › group (BIDS **chips** like
  `ses-001` `run-1` `task-rest`; subject-level anat → `subject-level`) › T1w/T2w (uppercase
  **eyebrow**) › figure (linked title + optional caption + **borderless** image). No
  indentation — levels read by treatment, not nesting depth.
- **Run status** is calm: neutral card, ink headline, colour only on a `Pass`/`Fail` badge
  and a thin left border.
- The top bar is a fixed flat bar; multi-group modalities use a native `<details>`
  dropdown closed by a tiny capture-phase script (no jQuery/Bootstrap).

## Components (see `styleguide.html`)

`.bn-card` · `.bn-callout` · `.bn-code` / `.bn-tag` · `.bn-btn` (`.ghost`) ·
`.bn-badge` (`.ok` / `.fail` / `.warn`) · `.bn-status` (`.ok` / `.fail`) ·
`.bn-h1` / `.bn-h2` underlines.
