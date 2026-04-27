import uproot
import awkward as ak
import numpy as np
import h5py
import argparse

# ── Maximum number of objects per event (pad/truncate to this) ──────────────
MAX_NTRACKS = 30
MAX_NPHOTONS = 35

# ── Feature definitions ─────────────────────────────────────────────────────
EVENT_FEATURES = ["event_p", "event_e", "event_n_charged", "event_n_neutral"]
TRACK_FEATURES = [
    "track_q", "track_p", "track_eta", "track_phi", "track_d0", "track_z0", "track_dndx",
    "track_prob_e", "track_prob_mu", "track_prob_pi", "track_prob_K", "track_prob_p",
]
PHOTON_FEATURES = ["photon_e", "photon_eta", "photon_phi"]

# ── Particle IDs for truth-matching ─────────────────────────────────────────
PHOTON_ID = 22
ELECTRON_ID = 11
MUON_ID = 13
KAON_ID = 321
PION_ID = 211
PROTON_ID = 2212

GLOB_VARS = ["EVT_p", "EVT_e", "EVT_nCharged", "EVT_nNeutral"]


def pad_and_flatten(jagged, features, max_n):
    """
    Pad/truncate a jagged record array to a fixed length and flatten.

    Returns a 2D numpy array of shape (n_events, max_n * n_features),
    with NaN for padded positions.  The memory layout per event is:
        [obj0_feat0, obj0_feat1, ..., obj1_feat0, obj1_feat1, ...]
    which can be reshaped to (max_n, n_features) by the model.
    """
    truncated = jagged[:, :max_n]
    padded = ak.pad_none(truncated, max_n, clip=True)

    columns = []
    for feat in features:
        col = ak.fill_none(padded[feat], np.nan)
        columns.append(ak.to_numpy(col).astype(np.float32))

    # (n_events, max_n, n_features) -> (n_events, max_n * n_features)
    return np.stack(columns, axis=-1).reshape(len(jagged), -1)


def process(input_file, output_file):
    """Read raw ROOT, process, pad, flatten, and save to HDF5."""

    tree = uproot.open(input_file + ":events")
    # Per-particle fields (charged + neutral)
    par = tree.arrays(["Rec_q", "Rec_p", "Rec_eta", "Rec_phi", "Rec_true_PDG"])
    # Track-specific fields (same per-particle ordering, 0 for neutrals)
    trk = tree.arrays(["Rec_track_d0", "Rec_track_z0", "Rec_track_dNdx"])

    # ── Photons ─────────────────────────────────────────────────────────────
    photon_mask = par["Rec_true_PDG"] == PHOTON_ID
    photons = ak.zip({
        "photon_e":   par["Rec_p"][photon_mask],
        "photon_eta": par["Rec_eta"][photon_mask],
        "photon_phi": par["Rec_phi"][photon_mask],
    })
    photons = photons[ak.argsort(photons["photon_e"], axis=1, ascending=False)]

    # ── Tracks ──────────────────────────────────────────────────────────────
    track_mask = par["Rec_q"] != 0
    absid = abs(par["Rec_true_PDG"][track_mask])
    tracks = ak.zip({
        "track_q":    par["Rec_q"][track_mask],
        "track_p":    par["Rec_p"][track_mask],
        "track_eta":  par["Rec_eta"][track_mask],
        "track_phi":  par["Rec_phi"][track_mask],
        "track_d0":   trk["Rec_track_d0"][track_mask],
        "track_z0":   trk["Rec_track_z0"][track_mask],
        "track_dndx": trk["Rec_track_dNdx"][track_mask],
    })
    tracks["track_prob_e"]  = ak.where(absid == ELECTRON_ID, 1, 0)
    tracks["track_prob_mu"] = ak.where(absid == MUON_ID, 1, 0)
    tracks["track_prob_pi"] = ak.where(absid == PION_ID, 1, 0)
    tracks["track_prob_K"]  = ak.where(absid == KAON_ID, 1, 0)
    tracks["track_prob_p"]  = ak.where(absid == PROTON_ID, 1, 0)
    tracks = tracks[ak.argsort(tracks["track_p"], axis=1, ascending=False)]
    # ── Event-level features ────────────────────────────────────────────────
    event = tree.arrays(GLOB_VARS)
    event_flat = np.column_stack([
        ak.to_numpy(event[v]).astype(np.float32) for v in GLOB_VARS
    ])

    # ── Target (random placeholder for now) ─────────────────────────────────
    target = tree.arrays(["MC_B_qTag"])
    target_flat = target["MC_B_qTag"].to_numpy().astype(np.int32)

    # ── Pad and flatten variable-length arrays ──────────────────────────────
    track_flat  = pad_and_flatten(tracks,  TRACK_FEATURES,  MAX_NTRACKS)
    photon_flat = pad_and_flatten(photons, PHOTON_FEATURES, MAX_NPHOTONS)

    # ── Concatenate: event | tracks | photons ───────────────────────────────
    X = np.concatenate([event_flat, track_flat, photon_flat], axis=1)
    y = target_flat

    # ── Shuffle ─────────────────────────────────────────────────────────────
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # ── Build ordered feature name list ─────────────────────────────────────
    feature_names = list(EVENT_FEATURES)
    for i in range(MAX_NTRACKS):
        feature_names += [f"{f}_{i}" for f in TRACK_FEATURES]
    for i in range(MAX_NPHOTONS):
        feature_names += [f"{f}_{i}" for f in PHOTON_FEATURES]

    # ── Save to HDF5 ────────────────────────────────────────────────────────
    chunk_rows = min(10240, len(X))
    with h5py.File(output_file, "w", track_order=True) as hf:
        hf.create_dataset("X", data=X,
                          chunks=(chunk_rows, X.shape[1]),
                          compression="gzip", compression_opts=4)
        hf.create_dataset("y", data=y,
                          chunks=(chunk_rows,),
                          compression="gzip", compression_opts=4)

        # Metadata so the model knows the feature layout
        hf.attrs["n_events"]          = len(X)
        hf.attrs["n_features"]        = X.shape[1]
        hf.attrs["n_event_features"]  = len(EVENT_FEATURES)
        hf.attrs["n_track_features"]  = len(TRACK_FEATURES)
        hf.attrs["n_photon_features"] = len(PHOTON_FEATURES)
        hf.attrs["max_ntracks"]       = MAX_NTRACKS
        hf.attrs["max_nphotons"]      = MAX_NPHOTONS
        hf.attrs["event_features"]    = EVENT_FEATURES
        hf.attrs["track_features"]    = TRACK_FEATURES
        hf.attrs["photon_features"]   = PHOTON_FEATURES
        hf.attrs["feature_names"]     = feature_names
    
    # ── Save to root ────────────────────────────────────────────────────────
    branches = {name: X[:, i] for i, name in enumerate(feature_names)}
    branches["qTag"] = y
    with uproot.recreate(output_file.replace(".h5",".root")) as outf:
        outf["events"] = branches

    print(f"Saved {len(X)} events × {X.shape[1]} features to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", default="test_FT_tuple.root", help="Input tuple")
    parser.add_argument("-o", "--output", default="training_data.h5", help="Output training file")
    args = parser.parse_args()

    process(args.input, args.output)
