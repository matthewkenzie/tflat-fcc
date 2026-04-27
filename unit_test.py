#!/usr/bin/env python3
"""
Smoke test for the TFlat training pipeline.

Generates a tiny synthetic HDF5 file (no dependency on real data) whose shape
matches the feature layout defined in unit_test_config.yaml, then runs the full
build → compile → train loop through fitter.fit() and checks the key invariants.

Run with:
    conda run -n tflat pytest unit_test.py -v
"""

import os
import numpy as np
import h5py
import pytest
import keras

from fitter import fit
from utils import load_config
from model import get_tflat_model

# ── Constants ────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "unit_test_config.yaml")
N_EVENTS    = 256   # enough for a train/val split with batch_size=64


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
