"""Training-stack checks for the segmentation CNN (fastsurfer_nn).

Each test pins a bug found in the 2026-11 review (docs_temp/t2w_segmentation_training.md):
augmentation probabilities were ignored, cortex weights used sparse IDs on dense labels,
plot colours were shifted by one, the LR schedule counted iterations as epochs, resume
was off by two epochs, a failed pretrained load trained from scratch silently, and the
train/val split could leak an animal into both sets.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from fastsurfer_nn.atlas.atlas_manager import get_atlas_manager
from fastsurfer_nn.config.defaults import get_cfg_defaults


@pytest.fixture(scope="module")
def arm2():
    return get_atlas_manager("ARM2")


def _cfg(**data):
    cfg = get_cfg_defaults()
    cfg.MODEL.NUM_CLASSES = 71
    cfg.MODEL.NUM_FILTERS = 71
    cfg.DATA.CLASS_OPTIONS = ["ARM2"]
    cfg.DATA.PLANE = "coronal"
    cfg.DATA.PADDED_SIZE = 256
    cfg.DATA.SIZES = [256]
    for k, v in data.items():
        setattr(cfg.DATA, k, v)
    return cfg


# ----------------------------------------------------------------------------- sparse / dense
def test_dense_sparse_round_trip(arm2):
    for plane, n in (("coronal", 71), ("axial", 71), ("sagittal", 36)):
        d2s = arm2.dense_to_sparse(plane)
        assert len(d2s) == n == arm2.num_dense_classes(plane)
        assert d2s[0] == 0  # dense 0 is background
        names = arm2.dense_class_names(plane)
        assert len(names) == n and names[0] == "background"
    sparse = np.array([0, 2, 1002, -1, -1001, 502, 1502])
    dense = arm2.to_dense(sparse, "coronal")
    assert np.array_equal(arm2.dense_to_sparse("coronal")[dense], sparse)


def test_sagittal_merges_hemispheres(arm2):
    dense = arm2.to_dense(np.array([2, 1002, -1, -1001]), "sagittal")
    assert dense[0] == dense[1] != 0 and dense[2] == dense[3] != 0
    expand = arm2.get_sagittal_to_bilateral_expansion()
    assert len(expand) == 71 and expand[0] == 0 and max(expand) == 35


def test_cortex_weights_use_dense_cortex_classes(arm2):
    """The cortex set used for weighting must be dense indices of cortical labels only."""
    for plane in ("coronal", "sagittal"):
        cortex_dense = arm2.cortex_dense(plane)
        assert 0 not in cortex_dense
        d2s = arm2.dense_to_sparse(plane)
        cortex_sparse = arm2.get_atlas_config(plane).cortex_labels
        assert 0 not in cortex_sparse
        assert {int(d2s[d]) for d in cortex_dense} <= set(cortex_sparse)
        if plane == "coronal":
            assert len(cortex_dense) == len(cortex_sparse) == 34


def test_get_sparse_labels_is_not_dense_order(arm2):
    """Background is appended LAST in file order; dense order has it FIRST (old plot bug)."""
    assert arm2.get_sparse_labels()[-1] == 0
    assert arm2.get_dense_to_sparse_mapping()[0] == 0


# ----------------------------------------------------------------------------- augmentation
def _batch(b=8):
    torch.manual_seed(0)
    images = torch.rand(b, 7, 64, 64)
    labels = torch.randint(0, 71, (b, 64, 64))
    weights = torch.ones(b, 64, 64)
    scale = torch.full((b, 2), 2.0)
    return images, labels, weights, scale


def test_gpu_augment_respects_zero_probability():
    from fastsurfer_nn.data_loader.gpu_augment import GpuAugment

    cfg = _cfg(AUG=["Scaling", "Rotation", "Translation", "BiasField", "RGamma"])
    for k in ("Scaling", "Rotation", "Translation", "BiasField", "RGamma"):
        setattr(cfg.DATA.AUG_PROBABILITIES, k, 0.0)
    aug = GpuAugment(cfg, torch.Generator().manual_seed(0))
    im, lb, w, sc = _batch()
    out = aug(im.clone(), lb.clone(), w.clone(), sc.clone())
    assert torch.equal(out[0], im) and torch.equal(out[1], lb) and torch.equal(out[3], sc)


def test_gpu_augment_scaling_updates_scale_and_keeps_labels_integer():
    from fastsurfer_nn.data_loader.gpu_augment import GpuAugment

    cfg = _cfg(AUG=["Scaling"])
    cfg.DATA.AUG_PROBABILITIES.Scaling = 1.0
    cfg.DATA.AUG_SCALING.SCALES = (0.8, 1.2)
    aug = GpuAugment(cfg, torch.Generator().manual_seed(1))
    im, lb, w, sc = _batch()
    im2, lb2, w2, sc2 = aug(im.clone(), lb.clone(), w.clone(), sc.clone())
    ratio = sc2 / sc
    assert torch.all((ratio >= 0.8 - 1e-6) & (ratio <= 1.2 + 1e-6))
    assert not torch.allclose(ratio, torch.ones_like(ratio))
    assert lb2.dtype == torch.long and set(lb2.unique().tolist()) <= set(range(71))
    assert im2.min() >= 0 and im2.max() <= 1


def test_gpu_augment_probability_is_per_sample():
    from fastsurfer_nn.data_loader.gpu_augment import GpuAugment

    cfg = _cfg(AUG=["Scaling"])
    cfg.DATA.AUG_PROBABILITIES.Scaling = 0.5
    aug = GpuAugment(cfg, torch.Generator().manual_seed(2))
    _, lb, w, sc = _batch(400)
    im = torch.rand(400, 7, 8, 8)
    _, _, _, sc2 = aug(im, lb[:, :8, :8].clone(), w[:, :8, :8].clone(), sc)
    changed = (sc2[:, 0] != 2.0).float().mean().item()
    assert 0.4 < changed < 0.6


def test_gpu_augment_rejects_cpu_only_transforms():
    from fastsurfer_nn.data_loader.gpu_augment import GpuAugment

    with pytest.raises(ValueError, match="AUG_DEVICE"):
        GpuAugment(_cfg(AUG=["Elastic"]))


def test_cpu_transforms_carry_their_own_probability():
    """tio.Compose(dict) used to run every transform with p=1."""
    from fastsurfer_nn.data_loader.loader import build_cpu_transform

    cfg = _cfg(AUG=["Scaling", "BiasField", "RGamma"], AUG_DEVICE="cpu")
    cfg.DATA.AUG_PROBABILITIES.Scaling = 0.3
    cfg.DATA.AUG_PROBABILITIES.BiasField = 0.0
    cfg.DATA.AUG_PROBABILITIES.RGamma = 0.6
    tf = build_cpu_transform(cfg)
    probs = sorted(round(t.probability, 2) for t in tf.transforms)
    assert probs == [0.3, 0.6]  # BiasField (p=0) dropped


# ----------------------------------------------------------------------------- metrics
def test_dice_accumulator_matches_direct_computation():
    from fastsurfer_nn.utils.metrics import DiceAccumulator

    g = torch.Generator().manual_seed(3)
    pred = torch.randint(0, 5, (6, 16, 16), generator=g)
    ref = torch.randint(0, 5, (6, 16, 16), generator=g)
    subj = torch.tensor([0, 0, 0, 1, 1, 1])
    acc = DiceAccumulator(5, 2)
    acc.update(pred[:4], ref[:4], subj[:4])
    acc.update(pred[4:], ref[4:], subj[4:])
    sd = acc.subject_dice()
    for s in (0, 1):
        m = subj == s
        for c in range(5):
            p, r = pred[m] == c, ref[m] == c
            want = 2 * (p & r).sum() / (p.sum() + r.sum())
            assert math.isclose(sd[s, c].item(), want.item(), rel_tol=1e-6)
    per_subject = [np.mean([sd[s, c].item() for c in range(1, 5)]) for s in (0, 1)]
    assert math.isclose(acc.mean_subject_dice(), np.mean(per_subject), rel_tol=1e-6)


# ----------------------------------------------------------------------------- LR schedule
def test_lr_schedule_is_in_epochs():
    from fastsurfer_nn.utils.lr_scheduler import get_lr_scheduler

    cfg = _cfg()
    cfg.TRAIN.NUM_EPOCHS = 10
    cfg.OPTIMIZER.BASE_LR = 1e-3
    cfg.OPTIMIZER.ETA_MIN = 1e-5
    cfg.OPTIMIZER.LR_SCHEDULER = "cosine"
    cfg.OPTIMIZER.WARMUP_EPOCHS = 1.0
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    s = get_lr_scheduler(opt, cfg)
    assert math.isclose(s.get_last_lr()[0], 1e-4)  # warmup starts at 10%
    s.step(1.0)
    assert math.isclose(s.get_last_lr()[0], 1e-3)
    s.step(5.5)
    assert 1e-5 < s.get_last_lr()[0] < 1e-3
    s.step(10.0)
    assert math.isclose(s.get_last_lr()[0], 1e-5)

    cfg.OPTIMIZER.LR_SCHEDULER = "cosineWarmRestarts"
    cfg.OPTIMIZER.T_ZERO = 2
    cfg.OPTIMIZER.T_MULT = 2
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    s = get_lr_scheduler(opt, cfg)
    s.step(1.999)
    low = s.get_last_lr()[0]
    s.step(2.0)  # restart after T_ZERO EPOCHS (not iterations)
    assert low < 2e-5 and math.isclose(s.get_last_lr()[0], 1e-3)
    state = s.state_dict()
    s2 = get_lr_scheduler(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3), cfg)
    s2.load_state_dict(state)
    assert math.isclose(s2.get_last_lr()[0], s.get_last_lr()[0])


# ----------------------------------------------------------------------------- checkpoints
def test_checkpoint_round_trip_epoch_and_state(tmp_path):
    from fastsurfer_nn.utils import checkpoint as cp

    cfg = _cfg()
    cfg.DATA.PLANE = "sagittal"
    cfg.MODEL.NUM_CLASSES = 36
    cfg.DATA.CONTRAST = "T2w"
    model = torch.nn.Conv2d(1, 2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = {"best_epoch": 3, "wait_count": 2, "history": []}
    cp.save_checkpoint(tmp_path, 5, 0.81, 1, cfg, model, opt, extra_state=state,
                       provenance={"git_commit": "x"})
    paths = cp.get_checkpoint_path(tmp_path.parent / tmp_path.name / "..", None)
    ck = sorted(tmp_path.glob("Epoch_*"))[-1]
    epoch, best, st = cp.restore_model_state_from_checkpoint(ck, model, opt)
    assert (epoch, best, st["best_epoch"], st["wait_count"]) == (5, 0.81, 3, 2)
    meta = cp.extract_atlas_metadata(ck)
    assert meta["contrast"] == "T2w"
    assert len(meta["bilateral_dense_to_sagittal_dense"]) == 71
    assert meta["provenance"]["git_commit"] == "x"
    del paths


def test_legacy_checkpoint_epoch_is_corrected(tmp_path):
    from fastsurfer_nn.utils import checkpoint as cp

    model = torch.nn.Conv2d(1, 2, 1)
    torch.save({"model_state": model.state_dict(), "optimizer_state": None, "epoch": 6,
                "best_metric": 0.5, "config": ""}, tmp_path / "Epoch_00006_training_state.pkl")
    epoch, _, st = cp.restore_model_state_from_checkpoint(
        tmp_path / "Epoch_00006_training_state.pkl", model)
    assert epoch == 5 and st == {}  # old trainer saved epoch + 1


def test_get_checkpoint_path_finds_flat_checkpoints(tmp_path):
    from fastsurfer_nn.utils import checkpoint as cp

    (tmp_path / "checkpoints").mkdir()
    for e in (2, 10, 1):
        (tmp_path / "checkpoints" / f"Epoch_{e:05d}_training_state.pkl").write_bytes(b"")
    got = [p.name for p in cp.get_checkpoint_path(tmp_path)]
    assert got[-1] == "Epoch_00010_training_state.pkl" and len(got) == 3


def test_missing_pretrained_raises(tmp_path):
    from fastsurfer_nn.training.trainer import resolve_pretrained

    with pytest.raises(FileNotFoundError):
        resolve_pretrained(str(tmp_path / "nope" / "missing.pkl"))
    # a stale directory with a known file name falls back to PRETRAINED_MODEL_DIR
    assert resolve_pretrained("/old/place/T1w_seg-ARM2_coronal.pkl").exists()


# ----------------------------------------------------------------------------- split
def test_split_is_animal_level(tmp_path):
    from fastsurfer_nn.training.step1_split_data import split_data

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for a in range(10):
        for ses in (1, 2, 3):
            stem = f"site-x_sub-{a:03d}_ses-{ses}_T2w"
            (tmp_path / "images" / f"{stem}.nii.gz").write_bytes(b"")
            (tmp_path / "labels" / f"{stem}_atlas-ARM2.nii.gz").write_bytes(b"")
    train, val = split_data(tmp_path, val_split=0.2, seed=1)
    animal = lambda s: s.split("_ses-")[0]
    assert {animal(s) for s in train}.isdisjoint({animal(s) for s in val})
    assert len(val) == 6
    with pytest.raises(SystemExit):  # never silently replace an existing split
        split_data(tmp_path, val_split=0.2, seed=1)
    json.loads((tmp_path / "data_split.json").read_text())


def test_sagittal_expansion_uses_checkpoint_mapping():
    from fastsurfer_nn.data_loader.data_utils import map_prediction_sagittal2full

    pred = torch.arange(3).float().view(1, 3, 1, 1)
    out = map_prediction_sagittal2full(pred, num_classes=3, idx_list=[0, 2, 2, 1])
    assert out.flatten().tolist() == [0, 2, 2, 1]
    with pytest.raises(ValueError):
        map_prediction_sagittal2full(pred, num_classes=3)


def test_site_balance_weights_sqrt_shares():
    """Site share of draws ~ sqrt(n_subjects); within a site every slice equally likely."""
    from types import SimpleNamespace

    from fastsurfer_nn.data_loader.loader import site_balance_weights

    subjects = [f"site-big_sub-{i:02d}" for i in range(16)] + ["site-small_sub-01"]
    subject_of = np.repeat(np.arange(17), [10] * 16 + [40])  # small site: 1 subject, 40 slices
    ds = SimpleNamespace(subjects=subjects, subject_of=subject_of)
    w = site_balance_weights(ds, "sqrt")
    big, small = w[:160].sum(), w[160:].sum()
    assert math.isclose(big / small, 4.0, rel_tol=1e-9)  # sqrt(16) / sqrt(1)
    assert np.allclose(w[:160], w[0]) and np.allclose(w[160:], w[160])
    assert site_balance_weights(ds, "none") is None
