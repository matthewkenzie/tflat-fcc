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
    "track_q", "track_p", "track_eta", "track_phi", "track_d0", "track_z0",
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
    raw_arrays = tree.arrays(filter_name="Rec_*")
    particles = ak.zip({field: raw_arrays[field] for field in raw_arrays.fields})

    # ── Photons ─────────────────────────────────────────────────────────────
    photons = particles[particles["Rec_true_PDG"] == PHOTON_ID]
    photons = ak.zip({
        "photon_e":   photons["Rec_p"],
        "photon_eta": photons["Rec_eta"],
        "photon_phi": photons["Rec_phi"],
    })
    photons = photons[ak.argsort(photons["photon_e"], axis=1, ascending=False)]

    # ── Tracks ──────────────────────────────────────────────────────────────
    tracks = particles[particles["Rec_q"] != 0]
    absid = abs(tracks["Rec_true_PDG"])
    tracks = ak.zip({
        "track_q":   tracks["Rec_q"],
        "track_p":   tracks["Rec_p"],
        "track_eta": tracks["Rec_eta"],
        "track_phi": tracks["Rec_phi"],
        "track_d0":  tracks["Rec_track_d0"],
        "track_z0":  tracks["Rec_track_z0"],
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
    target = np.random.choice([0, 1], size=len(event)).astype(np.float32)

    # ── Pad and flatten variable-length arrays ──────────────────────────────
    track_flat  = pad_and_flatten(tracks,  TRACK_FEATURES,  MAX_NTRACKS)
    photon_flat = pad_and_flatten(photons, PHOTON_FEATURES, MAX_NPHOTONS)

    # ── Concatenate: event | tracks | photons ───────────────────────────────
    X = np.concatenate([event_flat, track_flat, photon_flat], axis=1)
    y = target

    # ── Shuffle ─────────────────────────────────────────────────────────────
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # ── Save to HDF5 ────────────────────────────────────────────────────────
    chunk_rows = min(10240, len(X))
    with h5py.File(output_file, "w", track_order=True) as hf:
        hf.create_dataset("X", data=X,
                          chunks=(chunk_rows, X.shape[1]),
                          compression="gzip", compression_opts=4)
        hf.create_dataset("y", data=y,
                          chunks=(chunk_rows,),
                          compression="gzip", compression_opts=4)
        # Build the full ordered list of column names
        feature_names = list(EVENT_FEATURES)
        for i in range(MAX_NTRACKS):
            feature_names += [f"{f}_{i}" for f in TRACK_FEATURES]
        for i in range(MAX_NPHOTONS):
            feature_names += [f"{f}_{i}" for f in PHOTON_FEATURES]

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

    print(f"Saved {len(X)} events × {X.shape[1]} features to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", default="test_FT_tuple.root", help="Input tuple")
    parser.add_argument("-o", "--output", default="training_data.h5", help="Output training file")
    args = parser.parse_args()

    process(args.input, args.output)
