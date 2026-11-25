# Segmentation CNN training: 2026-11 review, T2w model (developer notes)

How the ARM2 segmentation networks are trained, what was wrong with the training stack before
2026-11, and how the T2w model was made. For maintainers. User-facing: `docs/processing.rst`.

Code: `src/fastsurfer_nn/training/` (step1 split, step2 HDF5, step3 + `trainer.py`),
`data_loader/{dataset,loader,gpu_augment}.py`, `utils/{monitor,lr_scheduler,metrics,checkpoint}.py`,
`scripts/{run_training_pipeline.sh,make_training_yaml.py,evaluate_checkpoint.py,training_dashboard.py}`.
Training data: `macaque/monkey_unet_prep` (its README). Tests: `tests/test_fastsurfer_training.py`.

## 1. Terminology: sparse and dense labels

* **sparse** = atlas label IDs as stored in label NIfTIs and the ColorLUT (0, 2, 16, 1002, 502, -1, -1501 ...).
* **dense** = contiguous network class indices 0..N-1, 0 = background. ARM2: N = 71 bilateral
  (axial/coronal), 36 hemisphere-merged (sagittal, by `key_nohemi`).

Not the TensorFlow sense (sparse = integer class, dense = one-hot). The HDF5 `aseg_dataset`
holds dense labels; `AtlasConfig` region sets hold sparse IDs; `AtlasManager.to_dense /
dense_to_sparse / dense_ids_of / cortex_dense / dense_class_names(plane)` convert per plane.
The checkpoint key `dense_to_sparse_mapping` (bilateral) is unchanged; sagittal checkpoints
also store `bilateral_dense_to_sagittal_dense`, which inference now prefers over rebuilding
the mapping from the runtime roiinfo.

## 2. What was wrong (found 2026-11) and what changed

| Problem | Effect | Fix |
|---|---|---|
| `tio.Compose(dict)` | every augmentation ran with p = 1; `AUG_PROBABILITIES` ignored | per-transform `p=` (CPU path); GPU path samples per slice |
| cortex weight mask matched sparse cortex IDs against dense labels; `cortex_labels` contained 0 | "thin cortex" weights landed on background and dense classes 2/16/50 | `cortex_dense(plane)`, 0 removed |
| HDF5 stored the ORIGINAL header zooms (pre-conform, original axis order) | wrong VINN scale input for anisotropic sources (uwmadison 0.27×0.5, UNC 0.55×0.8) | conformed zooms |
| `PRETRAINED_MODEL` path stale (`brainana/fastsurfer_nn/...`); load error caught and logged | would train from random weights | load failure raises; bare names resolve in `pretrained_model/` |
| tensorboard not installed; `SummaryWriter=None` then called | training crashed at start | optional; monitor files are primary |
| step 1 commented out; step 2 used all subjects when no split | train = val (leak); split also by session, not animal | split mandatory, by animal, never silently replaced |
| `scheduler.step(epoch + it/len)` with `T_ZERO: 500` | T_ZERO counted in epochs: LR never restarted, stayed ~2e-3 | one fractional-epoch scheduler; new `cosine` (warmup + decay) |
| resume looked in `checkpoints/<EXPR_NUM>/`, epoch saved +1 and restored +1 | resume found nothing / skipped 2 epochs; early-stop state lost | flat folder, `checkpoint_format: 2`, trainer state saved |
| best checkpoint by val Dice, early stopping by val loss | two criteria disagree | one `SELECTION_METRIC` (default per-subject 3-D Dice) |
| val loss ~2× train loss | val batches are ordered slices of one subject: absent classes score Dice-loss ≈ 1 | not a bug of the loss; val loss is not comparable to train loss, so it no longer selects |
| plot LUT from `get_labels()` (background LAST) | all colours shifted by one class | dense→sparse per plane |
| HDF5 float32 + gzip, 8 GB chunk cache per worker × 12 workers | ~1.8 MB/slice, gzip CPU, page cache blow-up risk | uint8 / uint8 / float16, lzf, 1 slice per chunk (~0.32 MB/slice); 64 MB cache |
| `DiceScore` built two B×65536×C one-hots per iteration + 0.5 s background `.item()` thread | GPU time and syncs | `DiceAccumulator` (bincount) |

Not changed: network, loss (Dice + weighted CE), preprocessing (`conform`: LIA, finest voxel,
uint8 0-255 with the 2.5×p99 centre cap). The `channels_last` and `torch.compile` options exist
but are off (both slower here, §3).

## 3. Speed

The bottleneck was the data side: ~2.3 s/iteration (batch 16) in Dec 2025 runs, ~30 min per
epoch (744 iterations), from CPU torchio augmentation at p = 1 (bias field is expensive) and
gzip float32 HDF5 reads.

Profile on the new stack (A6000, batch 16, 256², coronal, `step3 --profile`):

| | s/iter | data wait | note |
|---|---|---|---|
| GPU augmentation, bf16 | 0.52 | 0.02 % | default |
| CPU torchio augmentation (fixed p), bf16 | 0.57 | 13 % | 3 workers |
| network only, fp32 | 0.69 | – | microbenchmark |
| network only, bf16 | 0.44 | – | |
| bf16 + channels_last | 0.51 | – | 71 filters (not a multiple of 8) |
| bf16 + torch.compile | 0.47 | – | recompiles in the interpolation layer |

The remaining time is the network itself (convolutions, maxout, batch norm at 256² with 71
filters); the architecture is fixed when fine-tuning. VINN groups samples by scale factor in
Python, so scale factors stay on the CPU (GPU tensors would sync on every comparison).

## 4. Recipe (fine-tuning, `scripts/make_training_yaml.py`)

MODEL and DATA.{PLANE, CLASS_OPTIONS, SIZES, PADDED_SIZE, PREPROCESSING} are copied from the
starting checkpoint; the rest:

* AdamW, LR 1e-3 (batch 16), 1-epoch warmup from 10 %, cosine to 1e-5 over 60 epochs.
* Augmentation on the GPU: scale 0.85–1.15 (p 0.8), rotation ±10° (0.5), translation ±15 px (0.8),
  bias field order 3, |c| ≤ 0.5 (0.6), gamma log ±0.2 (0.3), noise σ ≤ 0.03 (0.2).
* Selection and early stopping: mean per-subject foreground Dice, patience 10 after 5 epochs.
* bf16 autocast; loss in fp32.

## 5. T2w model

Data: T2w images from brainana runs are already resampled onto their T1w
(`space-T1w_desc-preproc_T2w`), so the QC'd T1w ARM2 labels are used as they are
(`monkey_unet_prep/4_build_modal_set.py`; QC overlays on the T2w also catch bad T2w→T1w
registrations).

### Label projection into T2w scanner space (2026-11-24)

Training T2w are the raw acquisitions (brainana `space-T2wScanner_T2w` = BIDS T2w). The label is
brainana's CNN ARM2 on the conformed T1w, projected conformed → T1w scanner (conform inverse:
FLIRT .mat, converted with `sitk_rigid_registration.fsl_mat_to_world_affine`) → T2w scanner
(`from-scanner_to-T2wScanner` .h5, brainana's ANTs rigid) in one nearest-neighbour resampling
(`macaque/monkey_unet_prep/4b_build_from_brainana.py`).

- A first version converted the FLIRT matrix with `c3d_affine_tool -fsl2ras`; with a scanner T1w
  of det < 0 and a conformed T1w of det > 0 that misplaced labels (mcgill 032207 brain Dice 0.815
  vs the correct route; amu 032214 ~1 voxel). `fsl_mat_to_world_affine` reproduces brainana's own
  `atlas_space-scanner` from `atlas_space-T1w` exactly (Dice 1.000), which is now a per-session
  self-check.
- brainana's T2w→T1w registration was checked and is fine: its `desc-T2w2T1w` QC is aligned, and in
  its conformed outputs the CNN label fits the T2w within 0–1 voxel (bright CSF shifts the T2w edge).

<!-- counts, split, results: filled in when training is done -->

## 6. Monitoring

`TrainingMonitor` writes `<OUTPUT_DIR>/monitor/{metrics.jsonl,status.json,snapshots/}`;
`python -m fastsurfer_nn.scripts.training_dashboard --root <training_output>` serves a live page
(port 8765, plotly from the installed package, offline) and `--export report.html` a static copy.
