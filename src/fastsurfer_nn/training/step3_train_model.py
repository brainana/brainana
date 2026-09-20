# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import sys

# IMPORTS
from os.path import join

from fastsurfer_nn.training.trainer import Trainer
from fastsurfer_nn.utils import misc
from fastsurfer_nn.utils.constants import REPO_ROOT
from fastsurfer_nn.utils.load_config import get_config


def make_parser() -> argparse.ArgumentParser:
    """
    Set up the options parsed from STDIN.

    Parses arguments from the STDIN, including the flags: --cfg, --aug, --opt, opts.

    Returns
    -------
    argparse.ArgumentParser
        The parser object for options.
    """
    parser = argparse.ArgumentParser(description="Segmentation")

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=REPO_ROOT / "src/fastsurfer_nn/config/T1w_ARM2_coronal_lia.yaml",
        type=str,
    )
    parser.add_argument(
        "--aug", action="append", help="List of augmentations to use.", default=None
    )

    # Multi-view prediction plane weights
    parser.add_argument(
        "--plane-weight-coronal",
        dest="plane_weight_coronal",
        type=float,
        help="Weight for coronal plane in multi-view prediction (default: 0.4)",
        default=None,
    )
    parser.add_argument(
        "--plane-weight-axial",
        dest="plane_weight_axial",
        type=float,
        help="Weight for axial plane in multi-view prediction (default: 0.4)",
        default=None,
    )
    parser.add_argument(
        "--plane-weight-sagittal",
        dest="plane_weight_sagittal",
        type=float,
        help="Weight for sagittal plane in multi-view prediction (default: 0.2)",
        default=None,
    )

    parser.add_argument(
        "opts",
        help="See fastsurfer_nn/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):
    """
    First sets variables and then runs the trainer model.
    """
    cfg = get_config(args)

    if args.aug is not None:
        cfg.DATA.AUG = args.aug

    # Set plane weights if provided via command line
    if args.plane_weight_coronal is not None:
        cfg.MULTIVIEW.PLANE_WEIGHTS.CORONAL = args.plane_weight_coronal
    if args.plane_weight_axial is not None:
        cfg.MULTIVIEW.PLANE_WEIGHTS.AXIAL = args.plane_weight_axial
    if args.plane_weight_sagittal is not None:
        cfg.MULTIVIEW.PLANE_WEIGHTS.SAGITTAL = args.plane_weight_sagittal

    # Use flat directory structure without EXPR_NUM subdirectories
    cfg.SUMMARY_PATH = misc.check_path(join(cfg.LOG_DIR, "training_summary"))
    cfg.CONFIG_LOG_PATH = misc.check_path(join(cfg.LOG_DIR, "config"))

    with open(join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
        json.dump(cfg, json_file, indent=2)

    trainer = Trainer(cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    parser = make_parser()
    if len(sys.argv) == 1:
        parser.print_help()
    args = parser.parse_args()
    main(args)
