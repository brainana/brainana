#!/usr/bin/env python3
"""
Collect the anatomical outputs of an earlier brainana run for a func-only run (--anat_from).

Invoked once from ``main.nf`` at run start, after the effective config is generated.
Writes ``<output_dir>/nextflow_reports/reused_anat.json`` (read by main.nf to build the
functional workflow's anatomical channels) and links the earlier fastsurfer subject dirs
into ``<output_dir>/fastsurfer`` for the tSNR surface projection.

Usage:
    python3 collect_reused_anat.py --anat-from DIR --output-dir DIR --config-file FILE

Exits 1 with a message on stderr when the earlier outputs cannot be reused.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src/ to path for nhp_mri_prep imports (nextflow_scripts/ -> nhp_mri_prep -> src)
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect reusable anatomical outputs")
    parser.add_argument("--anat-from", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()

    from nhp_mri_prep.utils.nextflow import load_config
    from nhp_mri_prep.utils.anat_reuse import (
        AnatReuseError,
        collect_anat_outputs,
        link_fastsurfer,
    )

    config = load_config(args.config_file)
    output_space = config.get("template", {}).get("output_space", "NMT2Sym:res-05")
    try:
        found = collect_anat_outputs(
            args.anat_from, output_space, output_dir=args.output_dir
        )
    except AnatReuseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.output_dir) / "nextflow_reports" / "reused_anat.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found, indent=2))
    linked = link_fastsurfer(
        args.anat_from, args.output_dir, [f["id"] for f in found["fastsurfer"]]
    )
    print(
        f"Reusing anatomy from {args.anat_from}: {len(found['anat'])} T1w session(s), "
        f"{len(found['fastsurfer'])} surface subject(s) ({len(linked)} linked)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
