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

import yacs.config

from fastsurfer_nn.config.defaults import get_cfg_defaults

# Import path resolution utilities
try:
    from fastsurfer_nn.utils.config_utils import get_paths_from_config

    HAS_PATH_UTILS = True
except ImportError:
    HAS_PATH_UTILS = False


def get_config(args: argparse.Namespace) -> yacs.config.CfgNode:
    """
    Given the arguments, load and initialize the configs.

    Parameters
    ----------
    args : argparse.Namespace
        Object holding args.

    Returns
    -------
    yacs.config.CfgNode
        Configuration node.

    """
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if getattr(args, "output_dir", None):
        cfg.OUTPUT_DIR = args.output_dir

    # Resolve paths from configuration
    cfg = _resolve_paths(cfg, args.cfg_file)

    # Don't append config filename to LOG_DIR - use flat structure
    # cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    # cfg.LOG_DIR = join(cfg.LOG_DIR, cfg_file_name)

    return cfg


def _resolve_paths(cfg: yacs.config.CfgNode, cfg_file: str) -> yacs.config.CfgNode:
    """
    Resolve paths from configuration using direct paths format.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Configuration node.
    cfg_file : str
        Path to config file.

    Returns
    -------
    yacs.config.CfgNode
        Configuration with resolved paths.
    """
    if not HAS_PATH_UTILS:
        return cfg

    # Check if using direct path format
    if not (hasattr(cfg, "TRAINING_DATA_DIR") and hasattr(cfg, "OUTPUT_DIR")):
        # No path resolution needed - using explicit paths
        return cfg

    if cfg.TRAINING_DATA_DIR == "" or cfg.OUTPUT_DIR == "":
        # Paths not specified, skip resolution
        return cfg

    try:
        # Convert YACS config to dict for path resolution
        # Include PREPROCESSING section so orientation can be used for HDF5 file naming
        data_dict = {
            "PLANE": cfg.DATA.PLANE,
            "PATH_HDF5_TRAIN": cfg.DATA.PATH_HDF5_TRAIN
            if cfg.DATA.PATH_HDF5_TRAIN
            else "",
            "PATH_HDF5_VAL": cfg.DATA.PATH_HDF5_VAL if cfg.DATA.PATH_HDF5_VAL else "",
            "CLASS_OPTIONS": cfg.DATA.CLASS_OPTIONS
            if hasattr(cfg.DATA, "CLASS_OPTIONS")
            else [],
        }

        # Add PREPROCESSING section if it exists (needed for orientation-based HDF5 file naming)
        if hasattr(cfg.DATA, "PREPROCESSING") and hasattr(
            cfg.DATA.PREPROCESSING, "ORIENTATION"
        ):
            data_dict["PREPROCESSING"] = {
                "ORIENTATION": cfg.DATA.PREPROCESSING.ORIENTATION,
            }

        cfg_dict = {
            "TRAINING_DATA_DIR": cfg.TRAINING_DATA_DIR,
            "OUTPUT_DIR": cfg.OUTPUT_DIR,
            "DATA": data_dict,
        }

        # Get resolved paths
        paths = get_paths_from_config(cfg_dict)

        # Update config with resolved paths
        cfg.DATA.PATH_HDF5_TRAIN = str(paths["train_hdf5"])
        cfg.DATA.PATH_HDF5_VAL = str(paths["val_hdf5"])
        cfg.LOG_DIR = str(paths["log_dir"])

    except Exception as e:
        print(f"Warning: Failed to resolve paths from configuration: {e}")
        print("Falling back to explicit paths in config.")

    return cfg


def load_config(cfg_file: str) -> yacs.config.CfgNode:
    """
    Load a yaml config file.

    Parameters
    ----------
    cfg_file : str
        Configuration filepath.

    Returns
    -------
    yacs.config.CfgNode
        Configuration node.
    """
    # setup base
    cfg = get_cfg_defaults()
    cfg.EXPR_NUM = "Default"
    cfg.SUMMARY_PATH = ""
    cfg.CONFIG_LOG_PATH = ""
    # Overwrite with stored arguments
    cfg.merge_from_file(cfg_file)

    # Resolve paths from configuration
    cfg = _resolve_paths(cfg, cfg_file)

    return cfg
