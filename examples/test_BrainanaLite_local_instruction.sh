#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Local launcher for BrainanaLite.ipynb
# ─────────────────────────────────────────────────────────────────────────────
#
# What this script does
#   Creates a throwaway Jupyter kernel (Python 3.11 + JupyterLab only).
#   The notebook itself installs brainana on first "Run All" into:
#     WORKING_DIR/brainana_lite_env/
#   That keeps brainana deps out of system Python — same idea as the notebook's
#   "Environment isolation guard" in the ENVIRONMENT SETUP cell.
#
# Before you run this script
#   1. Open BrainanaLite.ipynb and set WORKING_DIR in the USER SETTINGS cell.
#   2. Put one or more T1w volumes directly in WORKING_DIR (any filename; .nii or .nii.gz):
#        $WORKING_DIR/*.nii.gz
#
# In Jupyter Lab
#   Open BrainanaLite.ipynb → Run All. Inline QC images appear after each step.
#
# ─────────────────────────────────────────────────────────────────────────────

set -e  # stop on first failed command

# Run from this folder so .venv sits next to the notebook.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# brainana requires Python >= 3.11 (notebook checks this in the next cell after settings).
uv venv .venv --python 3.11
source .venv/bin/activate

# Jupyter only — do not pip-install brainana here; the notebook handles that via uv.
uv pip install jupyterlab

# Blocks until you quit Jupyter Lab; then optional cleanup below runs.
# source .venv/bin/activate again before running the notebook again.
source .venv/bin/activate
jupyter lab BrainanaLite.ipynb

# ── Optional cleanup (runs after you exit Jupyter Lab) ───────────────────────
# Removes only this Jupyter venv (.venv/). Pipeline outputs and the notebook-managed
# brainana env under WORKING_DIR/brainana_lite_env/ are untouched.
# Comment out the next two lines if you want to reuse .venv on the next launch.
deactivate
rm -rf .venv
