# %%
import glob
import logging
import os
import sys
from pathlib import Path

# Add src/ to path (scripts/dev/nhp_mriprep/ -> scripts/dev/ -> scripts/ -> repo root)
_src_dir = Path(__file__).resolve().parents[3] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
try:
    from nhp_mri_prep.quality_control.dataset_report import (
        DATASET_REPORT_NAME,
        write_dataset_report,
    )
    from nhp_mri_prep.quality_control.reports import (
        build_report_data,
        write_html_report,
    )
    from nhp_mri_prep.utils.nextflow import load_config
except ImportError:
    raise ImportError(
        "Failed to import report functions from nhp_mri_prep.quality_control"
    )

# %%
# Dataset to report on. Defaults to the QC report example bundled in the repo;
# override with argv[1] or BRAINANA_REPORT_DIR (the latter also works in Jupyter).
_repo_root = Path(__file__).resolve().parents[3]
_default_dataset = _repo_root / "docs" / "_static" / "QCreport_example"
dataset_dir = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("BRAINANA_REPORT_DIR", str(_default_dataset))
)

# %%
# get sub dir list
sublist = sorted(glob.glob(f"{dataset_dir}/sub-*"))
sublist = [Path(sub) for sub in sublist if Path(sub).is_dir()]

# %%
config = load_config(f"{dataset_dir}/nextflow_reports/config.yaml")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_report_only")

# Build every subject's report data once; both kinds of page render from it.
built = []
for sub in sublist:
    data = build_report_data(sub / "figures", sub.parent / f"{sub.name}.html", config, logger)
    built.append((sub.name, data))

# %%
# Two or more subjects also get all_subjects_report.html (same as the pipeline);
# subject pages link to it.
dataset_href = None
if len(built) > 1:
    write_dataset_report(built, Path(dataset_dir) / DATASET_REPORT_NAME, logger)
    dataset_href = DATASET_REPORT_NAME

for name, data in built:
    data["metadata"]["dataset_report_href"] = dataset_href
    write_html_report(data, Path(dataset_dir) / f"{name}.html", logger)
# %%
