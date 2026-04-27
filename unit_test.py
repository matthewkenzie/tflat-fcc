#!/usr/bin/env python3
# Run unit tests only:        conda run -n tflat pytest unit_test.py -v
# Run integration test only:  conda run -n tflat pytest unit_test.py -m integration -v
# Run everything:             conda run -n tflat pytest unit_test.py -v --run-integration
"""
Smoke test for the TFlat training pipeline.

Generates a tiny synthetic HDF5 file (no dependency on real data) whose shape
matches the feature layout defined in unit_test_config.yaml, then runs the full
build → compile → train loop through fitter.fit() and checks the key invariants.

Run with:
    conda run -n tflat pytest unit_test.py -v
"""

import json
import os
import numpy as np
import h5py
import pytest
import matplotlib
matplotlib.use("Agg")  # headless-safe before any pyplot import
import keras

from fitter import fit
from utils import load_config
from model import get_tflat_model

# ── Constants ─────────────────────────────────────────────────────────────
CONFIG_PATH      = os.path.join(os.path.dirname(__file__), "unit_test_config.yaml")
FULL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
ROOT_FILE        = os.path.join(os.path.dirname(__file__), "FCCee_FT_tuples", "Bd_test.root")
N_EVENTS    = 256   # enough for a train/val split with batch_size=64
N_FIXTURE   = 256   # events subsampled for the integration test
# Expected feature width from unit_test_config.yaml (dndx=False default)
# 4 event + 30*11 track + 35*3 photon
_p = load_config(CONFIG_PATH)["parameters"]
EXPECTED_FEATURES = (_p["num_evt"] * _p["num_evt_features"]
                     + _p["num_trk"] * _p["num_trk_features"]
                     + _p["num_photon"] * _p["num_photon_features"])


# ── Fixture ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_h5(tmp_path_factory):
    """
    Build a minimal HDF5 file with random features and binary labels whose
    shape exactly matches the architecture parameters in unit_test_config.yaml.
    NaN padding is injected into the tail of the track/photon slots to mimic
    real data where events have fewer objects than the maximum.
    """
    cfg = load_config(CONFIG_PATH)
    p   = cfg["parameters"]

    n_evt_feats    = p["num_evt"]    * p["num_evt_features"]
    n_trk_feats    = p["num_trk"]    * p["num_trk_features"]
    n_photon_feats = p["num_photon"] * p["num_photon_features"]
    n_features     = n_evt_feats + n_trk_feats + n_photon_feats

    rng = np.random.default_rng(seed=42)
    X   = rng.standard_normal((N_EVENTS, n_features)).astype(np.float32)
    y   = rng.integers(0, 2, size=N_EVENTS).astype(np.int32)

    # Inject NaN padding into the second half of track/photon slots
    trk_pad_start = n_evt_feats + (p["num_trk"] // 2) * p["num_trk_features"]
    X[:, trk_pad_start:n_evt_feats + n_trk_feats] = np.nan
    photon_pad_start = n_evt_feats + n_trk_feats + (p["num_photon"] // 2) * p["num_photon_features"]
    X[:, photon_pad_start:] = np.nan

    path = str(tmp_path_factory.mktemp("data") / "test_fixture.h5")
    chunk_rows = min(64, N_EVENTS)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("X", data=X, chunks=(chunk_rows, n_features))
        hf.create_dataset("y", data=y, chunks=(chunk_rows,))
        hf.attrs["n_events"]   = N_EVENTS
        hf.attrs["n_features"] = n_features
    return path


# ── Tests ────────────────────────────────────────────────────────────────────
def test_trainer_smoke(synthetic_h5, tmp_path):
    """
    Full pipeline smoke test: build → compile → train → checkpoint.
    Checks that:
      - training completes without exception
      - all reported losses are finite
      - the checkpoint file is written to disk
      - validation loss and accuracy keys are present in the history
    """
    cfg = load_config(CONFIG_PATH)
    # Speed overrides: 2 epochs, small batches, enough val events for ≥1 batch
    cfg["epochs"]               = 2
    cfg["batch_size"]           = 32
    cfg["chunk_size"]           = 128
    cfg["train_valid_fraction"] = 0.8   # 256×0.2 = 51 val events > batch_size

    checkpoint = str(tmp_path / "checkpoint.model.keras")

    model = get_tflat_model(parameters=cfg["parameters"])
    scheduler = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg["initial_learning_rate"],
        decay_steps=cfg["decay_steps"],
        alpha=cfg["alpha"],
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=scheduler,
            weight_decay=cfg["weight_decay"],
        ),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy", keras.metrics.AUC(), keras.metrics.MeanSquaredError()],
    )

    history = fit(model, synthetic_h5, cfg, checkpoint)
    hist    = history.history

    assert "loss"     in hist, "history missing 'loss'"
    assert "val_loss" in hist, "history missing 'val_loss'"
    assert "accuracy" in hist, "history missing 'accuracy'"
    assert all(np.isfinite(v) for v in hist["loss"]),     "training loss is not finite"
    assert all(np.isfinite(v) for v in hist["val_loss"]), "validation loss is not finite"
    assert os.path.isfile(checkpoint), "checkpoint was not saved"


def test_model_output_shape(synthetic_h5):
    """Model output should be (batch, 1) with values in (0, 1)."""
    cfg = load_config(CONFIG_PATH)
    model = get_tflat_model(parameters=cfg["parameters"])

    with h5py.File(synthetic_h5) as hf:
        X_batch = hf["X"][:32]

    preds = model.predict(X_batch, verbose=0)
    assert preds.shape == (32, 1), f"unexpected output shape {preds.shape}"
    assert np.all(preds >= 0) and np.all(preds <= 1), "sigmoid output out of [0, 1]"


# ── Integration test ─────────────────────────────────────────────────────────
def _train_cfg():
    """Return a fast training config for smoke tests (unit tests)."""
    cfg = load_config(CONFIG_PATH)
    cfg["epochs"]               = 2
    cfg["batch_size"]           = 32
    cfg["chunk_size"]           = 128
    cfg["train_valid_fraction"] = 0.8   # 256×0.2 = 51 val events > batch_size
    return cfg


def _slow_train_cfg():
    """Return a realistic training config targeting ~5 min on Bd_test.root.

    Uses the full config.yaml (embedding_dims=128) on all ~978 processed events.
    At ~13 s/epoch (14 batches of 64 on 900 train events) this runs for
    roughly 20 × 13 s ≈ 4.3 min of pure training.
    """
    cfg = load_config(FULL_CONFIG_PATH)
    cfg["epochs"]   = 20
    cfg["patience"] = 10   # allow early stopping if it genuinely converges
    return cfg


@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """
    End-to-end integration test:
      1. process()  : ROOT → HDF5  (checks shape, y values, ROOT mirror)
      2. make_fixture inside process() → fixture HDF5  (checks n_events, n_features)
      3. fit()      : train on fixture  (checks finite losses + checkpoint)
    Skipped automatically if FCCee_FT_tuples/Bd_test.root is not present.
    """
    if not os.path.exists(ROOT_FILE):
        pytest.skip(f"ROOT file not found: {ROOT_FILE}")

    from process      import process
    from make_fixture import make_fixture

    h5_out       = str(tmp_path / "training.h5")
    fixture_path = str(tmp_path / "fixture.h5")
    root_out     = h5_out.replace(".h5", ".root")

    # ── Step 1: process ROOT → HDF5 + fixture ─────────────────────────────
    process(ROOT_FILE, h5_out, fixture_path=fixture_path, fixture_events=N_FIXTURE)

    assert os.path.isfile(h5_out),   "HDF5 output not created"
    assert os.path.isfile(root_out), "ROOT mirror not created"
    with h5py.File(h5_out) as hf:
        n_events, n_features = hf["X"].shape
        assert n_events  > 0,                    "HDF5 has no events"
        assert n_features == EXPECTED_FEATURES,  \
            f"feature count mismatch: got {n_features}, expected {EXPECTED_FEATURES}"
        assert set(np.unique(hf["y"][:])).issubset({-1, 0, 1}), \
            "unexpected y values (expected B-meson qTag in {-1, 0, 1})"

    # ── Step 2: fixture ───────────────────────────────────────────────
    assert os.path.isfile(fixture_path), "fixture HDF5 not created"
    with h5py.File(fixture_path) as hf:
        fx_events, fx_features = hf["X"].shape
        assert fx_events   == min(N_FIXTURE, n_events), "fixture event count wrong"
        assert fx_features == EXPECTED_FEATURES,        "fixture feature count wrong"

    # ── Step 3: train on full processed data ──────────────────────────
    cfg        = _slow_train_cfg()
    checkpoint = str(tmp_path / "checkpoint.model.keras")

    model = get_tflat_model(parameters=cfg["parameters"])
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=cfg["initial_learning_rate"],
                decay_steps=cfg["decay_steps"],
                alpha=cfg["alpha"],
            ),
            weight_decay=cfg["weight_decay"],
        ),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy", keras.metrics.AUC(), keras.metrics.MeanSquaredError()],
    )

    history = fit(model, h5_out, cfg, checkpoint)
    hist    = history.history

    assert all(np.isfinite(v) for v in hist["loss"]),     "training loss is not finite"
    assert all(np.isfinite(v) for v in hist["val_loss"]), "val loss is not finite"
    assert os.path.isfile(checkpoint), "checkpoint was not saved"

    # ── Step 4: plot_history ────────────────────────────────────
    from plot_history import plot_history
    history_json = checkpoint.replace(".model.keras", ".history.json")
    history_png  = str(tmp_path / "history.png")
    assert os.path.isfile(history_json), "fitter did not save history JSON"
    plot_history(history_json, history_png)
    assert os.path.isfile(history_png), "plot_history did not produce PNG"

    # ── Step 5: plot_output ────────────────────────────────────
    from plot_output import plot_output
    output_png   = str(tmp_path / "output.png")
    metrics_json = output_png.rsplit(".", 1)[0] + ".json"
    plot_output(checkpoint, h5_out, FULL_CONFIG_PATH, output_png)
    assert os.path.isfile(output_png),   "plot_output did not produce PNG"
    assert os.path.isfile(metrics_json), "plot_output did not produce metrics JSON"
    with open(metrics_json) as f:
        metrics = json.load(f)
    assert {"train", "val"} <= metrics.keys(), "metrics JSON missing train/val keys"
    for split in ("train", "val"):
        for key in ("w", "D", "P"):
            assert np.isfinite(metrics[split][key]), \
                f"metrics[{split!r}][{key!r}] is not finite"
