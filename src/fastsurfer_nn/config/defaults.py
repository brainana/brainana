from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Path Management
# ---------------------------------------------------------------------------- #
# Direct path to training data directory (contains HDF5 files)
_C.TRAINING_DATA_DIR = ""

# Direct path to output directory (for training logs, checkpoints, etc.)
_C.OUTPUT_DIR = ""

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = "FastSurferVINN"

# Number of classes to predict, including background
# This value is atlas-agnostic and must be set via config file or checkpoint
# Different atlases have different numbers of classes (e.g., ARM2=71, ARM3 may differ)
_C.MODEL.NUM_CLASSES = 0  # Placeholder: must be overridden by config or checkpoint

# Loss function, combined = dice loss + cross entropy, combined2 = dice loss + boundary loss
_C.MODEL.LOSS_FUNC = "combined"

# Filter dimensions for DenseNet (all layers same)
# This value is atlas-agnostic and must be set via config file or checkpoint
# Often matches NUM_CLASSES but can vary (e.g., ARM2=71, binary=2)
_C.MODEL.NUM_FILTERS = 0  # Placeholder: must be overridden by config or checkpoint

# Filter dimensions for Input Interpolation block (currently all the same)
_C.MODEL.NUM_FILTERS_INTERPOL = 32

# Number of UNet layers in Basenetwork (including bottleneck layer!)
_C.MODEL.NUM_BLOCKS = 5

# Number of input channels (slice thickness)
_C.MODEL.NUM_CHANNELS = 7

# Height of convolution kernels
_C.MODEL.KERNEL_H = 3

# Width of convolution kernels
_C.MODEL.KERNEL_W = 3

# size of Classifier kernel
_C.MODEL.KERNEL_C = 1

# Stride during convolution
_C.MODEL.STRIDE_CONV = 1

# Stride during pooling
_C.MODEL.STRIDE_POOL = 2

# Size of pooling filter
_C.MODEL.POOL = 2

# The height of segmentation model (after interpolation layer)
_C.MODEL.HEIGHT = 256

# The width of segmentation model
_C.MODEL.WIDTH = 256

# Interpolation mode for up/downsampling in Flex networks
_C.MODEL.INTERPOLATION_MODE = "bilinear"

# Crop positions for up/downsampling in Flex networks
_C.MODEL.CROP_POSITION = "top_left"

# Out Tensor dimensions for interpolation layer
_C.MODEL.OUT_TENSOR_WIDTH = 320
_C.MODEL.OUT_TENSOR_HEIGHT = 320

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# input batch size for training
_C.TRAIN.BATCH_SIZE = 16

# how many batches to wait before logging training status
_C.TRAIN.LOG_INTERVAL = 50

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.RESUME = False

# The experiment number to resume from
_C.TRAIN.RESUME_EXPR_NUM = "Default"

# number of epochs to train
_C.TRAIN.NUM_EPOCHS = 30

# number of steps (iteration) which depends on dataset
_C.TRAIN.NUM_STEPS = 10

# To fine tune model or not
_C.TRAIN.FINE_TUNE = False

# Path to pretrained model checkpoint for transfer learning
_C.TRAIN.PRETRAINED_MODEL = ""

# checkpoint period
_C.TRAIN.CHECKPOINT_PERIOD = 2

# number of worker for dataloader
_C.TRAIN.NUM_WORKERS = 8

# Flag to disable or enable Early Stopping
_C.TRAIN.EARLY_STOPPING = True

# Mode for early stopping (min = stop when metric is no longer decreasing,
# max = stop when mwtric is no longer increasing)
_C.TRAIN.EARLY_STOPPING_MODE = "min"

# Patience = Number of epochs to wait before stopping
_C.TRAIN.EARLY_STOPPING_PATIENCE = 10

# Wait = NUmber of epochs before starting early stopping check
_C.TRAIN.EARLY_STOPPING_WAIT = 10

# Delta = change below which early stopping starts (previous - current < delta = stop)
_C.TRAIN.EARLY_STOPPING_DELTA = 0.00001

# Metric that selects the best checkpoint AND drives early stopping (one criterion):
# "val_dice" (per-subject 3-D foreground Dice on the validation set, higher is better) or
# "val_loss" (lower is better). EARLY_STOPPING_MODE is derived from it.
_C.TRAIN.SELECTION_METRIC = "val_dice"

# Mixed precision: "bf16" (Ampere+ GPUs), "fp16" (uses a grad scaler) or "none"
_C.TRAIN.AMP = "bf16"

# channels_last memory format. Off: with 71 filters (not a multiple of 8) it was 15% SLOWER
# on an A6000 (2026-11 benchmark, bf16: 0.44 vs 0.51 s/iter).
_C.TRAIN.CHANNELS_LAST = False

# torch.compile the model (first epoch is slower; off by default)
_C.TRAIN.COMPILE = False

# Deterministic cuDNN (slower). When False, cudnn.benchmark is enabled.
_C.TRAIN.DETERMINISTIC = False

# Gradient accumulation: optimizer step every N batches (effective batch = N * BATCH_SIZE)
_C.TRAIN.ACCUM_STEPS = 1

# Number of fixed validation slices drawn in the per-epoch snapshot (monitor/snapshots)
_C.TRAIN.SNAPSHOT_SLICES = 4

# Write the live-monitor files (monitor/metrics.jsonl, status.json, snapshots)
_C.TRAIN.MONITOR = True

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# input batch size for testing
_C.TEST.BATCH_SIZE = 16

# Edge padding percentage (0.0 to 1.0) - inference only
# Adds symmetric padding on each edge after conforming to help model recognize
# brain tissue near image boundaries. Example: 0.05 = 5% padding on each edge
# (total 10% increase per dimension). Default: 0.0 (no padding)
_C.TEST.EDGE_PADDING_PERCENT = 0.00

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #

_C.DATA = CN()

# path to training hdf5-dataset
_C.DATA.PATH_HDF5_TRAIN = ""

# path to validation hdf5-dataset
_C.DATA.PATH_HDF5_VAL = ""

# The plane to load ['axial', 'coronal', 'sagittal', 'mixed']
# 'mixed' mode: processes all 3 planes per subject for plane-agnostic training
_C.DATA.PLANE = "coronal"

# Which classes to use
_C.DATA.CLASS_OPTIONS = ["aseg", "aparc"]

# Available size for dataloader
# This for the multi-scale dataloader
_C.DATA.SIZES = [256, 311, 320]

# the size that all inputs are padded to
_C.DATA.PADDED_SIZE = 320

# Image contrast the model is trained for ("T1w", "T2w", "T2wExvivo"); stored in the
# checkpoint metadata so the pipeline can check it picked the right model.
_C.DATA.CONTRAST = ""

# Augmentations. Geometric: Scaling, Rotation, Translation (one combined affine), Elastic,
# RAnisotropy; intensity: BiasField, RGamma, Noise. "Gaussian" is NOT image noise: it
# jitters the network's scale-factor input (FastSurfer legacy). Use "Noise" for image noise.
_C.DATA.AUG = ["Scaling", "Translation"]

# Where augmentation runs: "gpu" (batched, in the trainer; supports Scaling, Rotation,
# Translation, BiasField, RGamma, Noise, Gaussian) or "cpu" (torchio in the data loader
# workers; also supports Elastic and RAnisotropy, but is much slower).
_C.DATA.AUG_DEVICE = "gpu"

# Site balancing of the TRAINING draws (validation is never weighted). Subjects are grouped by
# the "site-<name>_" prefix of their name. "none": uniform over slices; "sqrt": each site's
# share of draws is proportional to sqrt(number of its subjects); "equal": every site the same.
_C.DATA.SITE_BALANCE = "none"

# Individual probabilities for each augmentation (0.0 to 1.0)
# If not specified in config, defaults to 0.8 for each augmentation
_C.DATA.AUG_PROBABILITIES = CN()
_C.DATA.AUG_PROBABILITIES.Rotation = 0.8
_C.DATA.AUG_PROBABILITIES.Scaling = 0.8
_C.DATA.AUG_PROBABILITIES.Translation = 0.8
_C.DATA.AUG_PROBABILITIES.BiasField = 0.8
_C.DATA.AUG_PROBABILITIES.Gaussian = 0.8
_C.DATA.AUG_PROBABILITIES.Elastic = 0.8
_C.DATA.AUG_PROBABILITIES.RAnisotropy = 0.8
_C.DATA.AUG_PROBABILITIES.RGamma = 0.8
_C.DATA.AUG_PROBABILITIES.Noise = 0.0

# Image noise augmentation: additive Gaussian, std drawn from U(0, STD) (image in [0, 1])
_C.DATA.AUG_NOISE = CN()
_C.DATA.AUG_NOISE.STD = 0.03

# Elastic deformation parameters (for performance tuning)
# Reducing NUM_CONTROL_POINTS from 7 to 5 significantly speeds up Elastic transform
# while maintaining reasonable deformation quality
# Note: With locked_borders=2, torchio requires at least 5 control points
_C.DATA.AUG_ELASTIC = CN()
_C.DATA.AUG_ELASTIC.NUM_CONTROL_POINTS = (
    5  # Reduced from 7 for better performance (minimum: 5 with locked_borders=2)
)
_C.DATA.AUG_ELASTIC.MAX_DISPLACEMENT = (20, 20, 0)  # Max displacement in pixels

# Scaling augmentation parameters
_C.DATA.AUG_SCALING = CN()
_C.DATA.AUG_SCALING.SCALES = (
    0.8,
    1.15,
)  # Scaling range (min, max) - isotropic scaling factor

# Rotation augmentation parameters
_C.DATA.AUG_ROTATION = CN()
_C.DATA.AUG_ROTATION.DEGREES = (
    5  # Random rotation range: ±degrees (e.g., 5 means -5 to +5 degrees)
)

# Translation augmentation parameters
_C.DATA.AUG_TRANSLATION = CN()
_C.DATA.AUG_TRANSLATION.TRANSLATION = (
    15.0,
    15.0,
    0,
)  # Translation range in pixels (x, y, z)

# Random Anisotropy augmentation parameters
_C.DATA.AUG_RANISOTROPY = CN()
_C.DATA.AUG_RANISOTROPY.AXES = (0, 1)  # Axes along which to apply anisotropy
_C.DATA.AUG_RANISOTROPY.DOWNSAMPLING = (
    1.1,
    1.3,
)  # Downsampling range (min, max) - reduced from (1.1, 1.5) for better performance

# Bias Field augmentation parameters
_C.DATA.AUG_BIASFIELD = CN()
_C.DATA.AUG_BIASFIELD.COEFFICIENTS = (
    0.3,
    0.7,
)  # Bias field coefficient range (min, max)
_C.DATA.AUG_BIASFIELD.ORDER = (
    2  # Polynomial order for bias field (reduced from 3 to 2 for better performance)
)

# Random Gamma augmentation parameters
_C.DATA.AUG_RGAMMA = CN()
_C.DATA.AUG_RGAMMA.LOG_GAMMA = (
    -0.1,
    0.1,
)  # Log gamma range (min, max) for gamma correction

# ---------------------------------------------------------------------------- #
# Preprocessing options - Single Source of Truth for HDF5, Training & Inference
# ---------------------------------------------------------------------------- #
_C.DATA.PREPROCESSING = CN()

# Target orientation: 3-letter code
# Common orientations: RAS (Right-Anterior-Superior), LIA (Left-Inferior-Anterior)
_C.DATA.PREPROCESSING.ORIENTATION = "RAS"

# Image size: int (e.g. 256), "fov" (preserve field of view), or "cube" (make cube)
#   - "fov" (RECOMMENDED): Preserves exact physical FOV, brain position maintained
#     Calculates: target_size = ceil(original_FOV / target_vox_size)
#     Result: May have different dimensions per axis (e.g., [320, 144, 210])
#     Use when: You want to preserve brain position exactly (no shifting)
#   - "cube": First calculates FOV-based size, then pads to cube using max dimension
#     Result: Same size in all axes (e.g., [320, 320, 320])
#     Preserves brain position by padding after affine calculation
#     Use when: You need a cube for model compatibility
#   - int: Forces a cube of that size (e.g., 256 → [256, 256, 256])
_C.DATA.PREPROCESSING.IMG_SIZE = "fov"

# Voxel size: float (e.g. 1.0) or "min" (adaptive)
_C.DATA.PREPROCESSING.VOX_SIZE = "min"

# Output data type for images: "uint8", "float32", etc.
_C.DATA.PREPROCESSING.DTYPE_IMAGE = "uint8"

# Output data type for labels: "int16" or "int32" (supports negative IDs!)
_C.DATA.PREPROCESSING.DTYPE_LABEL = "int16"

# Rescale image intensity to [0, RESCALE]
_C.DATA.PREPROCESSING.RESCALE = 255

# Interpolation order for images (0=nearest, 1=linear, 3=cubic)
_C.DATA.PREPROCESSING.ORDER_IMAGE = 1

# Interpolation order for labels (always 0=nearest neighbor)
_C.DATA.PREPROCESSING.ORDER_LABEL = 0

# ---------------------------------------------------------------------------- #
# DataLoader options (common for test and train)
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers
_C.DATA_LOADER.NUM_WORKERS = 8

# Prefetch factor: number of batches each worker prepares ahead
# Higher values (4-8) improve GPU utilization but use more memory
# Lower values (1-2) use less memory but may cause GPU to wait for data
_C.DATA_LOADER.PREFETCH_FACTOR = 4

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Validation loader workers (validation is not augmented; 1 is usually enough)
_C.DATA_LOADER.VAL_NUM_WORKERS = 1

# HDF5 chunk cache per open file handle (each worker opens its own handle)
_C.DATA_LOADER.HDF5_CACHE_MB = 64

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

# Base learning rate.
_C.OPTIMIZER.BASE_LR = 0.01

# Learning rate scheduler: "cosine" (linear warmup for WARMUP_EPOCHS, then cosine decay to
# ETA_MIN at NUM_EPOCHS), "cosineWarmRestarts" (T_ZERO/T_MULT in EPOCHS), "step_lr" or
# "NoScheduler". All schedulers are stepped every optimizer step with fractional epochs.
_C.OPTIMIZER.LR_SCHEDULER = "cosineWarmRestarts"

# Linear warmup length in epochs (fractional allowed); used by "cosine"
_C.OPTIMIZER.WARMUP_EPOCHS = 0.0

# Multiplicative factor of learning rate decay in step_lr
_C.OPTIMIZER.GAMMA = 0.3

# Period of learning rate decay in step_lr
_C.OPTIMIZER.STEP_SIZE = 5

# minimum learning in cosine lr policy and reduceLROnPlateau
_C.OPTIMIZER.ETA_MIN = 0.0001

# length of the first cosine cycle in cosineWarmRestarts, in EPOCHS
_C.OPTIMIZER.T_ZERO = 10

# A factor increases T_i after a restart in cosineWarmRestarts
_C.OPTIMIZER.T_MULT = 2

# FACTOR/PATIENCE/THRESH/COOLDOWN are unused (no plateau scheduler); kept so configs
# embedded in older checkpoints still load. MOMENTUM/DAMPENING/NESTEROV apply to sgd only.
_C.OPTIMIZER.FACTOR = 0.1

# number of epochs to wait before lowering lr (default=5)
_C.OPTIMIZER.PATIENCE = 5

# Threshold for measuring new optimum (default=1e-4)
_C.OPTIMIZER.THRESH = 0.0001

# Number of epochs to wait before resuming normal operation (default=0)
_C.OPTIMIZER.COOLDOWN = 0

# Momentum
_C.OPTIMIZER.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIMIZER.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIMIZER.NESTEROV = True

# L2 regularization
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

# Optimization method [sgd, adam, adamW]
_C.OPTIMIZER.OPTIMIZING_METHOD = "adamW"

# ---------------------------------------------------------------------------- #
# Multi-view prediction options
# ---------------------------------------------------------------------------- #
_C.MULTIVIEW = CN()

# Plane weights for multi-view prediction (coronal, axial, sagittal)
# Default weights: coronal=0.4, axial=0.4, sagittal=0.2
_C.MULTIVIEW.PLANE_WEIGHTS = CN()
_C.MULTIVIEW.PLANE_WEIGHTS.CORONAL = 0.4
_C.MULTIVIEW.PLANE_WEIGHTS.AXIAL = 0.4
_C.MULTIVIEW.PLANE_WEIGHTS.SAGITTAL = 0.2

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use
_C.NUM_GPUS = 1

# log directory for run
_C.LOG_DIR = "./experiments"

# experiment number
_C.EXPR_NUM = "Default"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

_C.SUMMARY_PATH = "FastSurferVINN/summary/FastSurferVINN_coronal"
_C.CONFIG_LOG_PATH = "FastSurferVINN/config/FastSurferVINN_coronal"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
