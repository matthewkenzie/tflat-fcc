import os
import sys
import importlib.util
import uproot
from make_fixture import make_fixture
import awkward as ak
import numpy as np
import h5py
import argparse

# Path to the fccee-tracker-pid package (sibling directory)
PID_PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fccee-tracker-pid")

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

def _load_pid_tools():
    """Import helper functions from fccee-tracker-pid without triggering its CLI entry point.

    get_pid.py runs argparse + assertions at module level, so we load it via
    importlib and catch the SystemExit/AssertionError that fires before the
    tree-reading code, by which point the three functions are already defined.
    """
    spec = importlib.util.spec_from_file_location(
        "pid_tools", os.path.join(PID_PACKAGE, "get_pid.py"))
    mod = importlib.util.module_from_spec(spec)
    import io
    _old_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        spec.loader.exec_module(mod)
    except (SystemExit, AssertionError):
        pass  # functions are defined before argparse/assertions run
    finally:
        sys.stderr = _old_stderr
    return mod.read_classifiers, mod.make_prediction, mod.get_features


def get_pid(tracks, absid, detector="IDEA", tof_val=-1.0, dndx_val=0.8):
    """Apply BDT-based PID from fccee-tracker-pid for pi/K/p; truth-match e/mu."""
    read_classifiers, make_prediction, get_features = _load_pid_tools()
    usededx = "nodedx"

    # Model paths inside the package are relative — change CWD temporarily
    orig_dir = os.getcwd()
    os.chdir(PID_PACKAGE)
    model_cache = read_classifiers(detector, tof_val, dndx_val, usededx)
    os.chdir(orig_dir)

    # Flatten jagged track arrays to 1D for the BDT
    counts = ak.num(tracks["track_p"])
    flat_data = {
        "track_p":    ak.to_numpy(ak.flatten(tracks["track_p"])).astype(np.float64),
        "track_dndx": ak.to_numpy(ak.flatten(tracks["track_dndx"])).astype(np.float64),
    }
    features = get_features(
        flat_data,
        speed_var=None, flight_var=None, tof_var=None,
        dndx_var="track_dndx", dedx_var=None,
        momentum_var="track_p", detector=detector,
    )

    # Predict probabilities for pi (211), K (321), p (2212)
    pdgs = [211, 321, 2212]
    predictions = make_prediction(model_cache, features, tof_val, dndx_val, usededx, pdgs)

    # Unflatten predictions back to jagged and assign to track record
    tracks["track_prob_pi"] = ak.unflatten(predictions[0].astype(np.float32), counts)
    tracks["track_prob_K"]  = ak.unflatten(predictions[1].astype(np.float32), counts)
    tracks["track_prob_p"]  = ak.unflatten(predictions[2].astype(np.float32), counts)
    # e/mu are not predicted by this tool — fall back to truth matching
    tracks["track_prob_e"]  = ak.where(absid == ELECTRON_ID, 1.0, 0.0)
    tracks["track_prob_mu"] = ak.where(absid == MUON_ID,     1.0, 0.0)
    return tracks

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


def process(input_file, output_file, dndx=False, getpid=False,
            fixture_path=None, fixture_events=512, max_events=None):
    """Read raw ROOT, process, pad, flatten, and save to HDF5."""

    tree = uproot.open(input_file + ":events")
    kw   = dict(entry_stop=max_events)   # passed to every tree.arrays() call
    # Per-particle fields (charged + neutral)
    par = tree.arrays(["Rec_q", "Rec_p", "Rec_eta", "Rec_phi", "Rec_true_PDG"], **kw)
    # Track-specific fields (same per-particle ordering, 0 for neutrals)
    trk = tree.arrays(["Rec_track_d0", "Rec_track_z0", "Rec_track_dNdx"], **kw)

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

    if getpid:
        tracks = get_pid(tracks, absid)
    else:
        tracks["track_prob_e"]  = ak.where(absid == ELECTRON_ID, 1, 0)
        tracks["track_prob_mu"] = ak.where(absid == MUON_ID, 1, 0)
        tracks["track_prob_pi"] = ak.where(absid == PION_ID, 1, 0)
        tracks["track_prob_K"]  = ak.where(absid == KAON_ID, 1, 0)
        tracks["track_prob_p"]  = ak.where(absid == PROTON_ID, 1, 0)
    tracks = tracks[ak.argsort(tracks["track_p"], axis=1, ascending=False)]
    


    # ── Event-level features ────────────────────────────────────────────────
    event = tree.arrays(GLOB_VARS, **kw)
    event_flat = np.column_stack([
        ak.to_numpy(event[v]).astype(np.float32) for v in GLOB_VARS
    ])

    # ── Target  ─────────────────────────────────────────────────────────────
    target = tree.arrays(["MC_B_qTag"], **kw)
    target_flat = target["MC_B_qTag"].to_numpy().astype(np.int32)

    # ── Pad and flatten variable-length arrays ────────────────────────────────────
    if not dndx:
        trk_feats = [t for t in TRACK_FEATURES if t != "track_dndx"]
    else:
        trk_feats = TRACK_FEATURES
    track_flat  = pad_and_flatten(tracks,  trk_feats,  MAX_NTRACKS)
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
        feature_names += [f"{f}_{i}" for f in trk_feats]
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
        hf.attrs["n_track_features"]  = len(trk_feats)
        hf.attrs["n_photon_features"] = len(PHOTON_FEATURES)
        hf.attrs["max_ntracks"]       = MAX_NTRACKS
        hf.attrs["max_nphotons"]      = MAX_NPHOTONS
        hf.attrs["event_features"]    = EVENT_FEATURES
        hf.attrs["track_features"]    = trk_feats 
        hf.attrs["photon_features"]   = PHOTON_FEATURES
        hf.attrs["feature_names"]     = feature_names
    
    # ── Save to root ────────────────────────────────────────────────────────
    branches = {name: X[:, i] for i, name in enumerate(feature_names)}
    branches["qTag"] = y
    with uproot.recreate(output_file.replace(".h5",".root")) as outf:
        outf["training"] = branches

    print(f"Saved {len(X)} events × {X.shape[1]} features to {output_file}")

    # ── Optionally write a small fixture for unit / integration tests ────────
    if fixture_path:
        make_fixture(output_file, fixture_path, fixture_events)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", default="test_FT_tuple.root", help="Input tuple")
    parser.add_argument("-o", "--output", default="training_data.h5", help="Output training file")
    parser.add_argument("-s", "--save-dndx", default=False, action="store_true", help="Save dNdx per track")
    parser.add_argument("-p", "--get-pid", default=False, action="store_true", help="Look for PID tool and gen PID vars")
    parser.add_argument("--fixture", default=None, metavar="PATH",
                        help="Also write a small test fixture HDF5 alongside the main output")
    parser.add_argument("--fixture-events", type=int, default=512, metavar="N",
                        help="Number of events to include in the fixture")
    parser.add_argument("--max-events", type=int, default=None, metavar="N",
                        help="Stop after reading this many events (useful for testing)")
    args = parser.parse_args()

    if args.save_dndx:
        args.output = args.output.replace(".h5", "_dNdx.h5")
    if args.get_pid:
        args.output = args.output.replace(".h5","_pid.h5")

    process(args.input, args.output, args.save_dndx, args.get_pid,
            fixture_path=args.fixture, fixture_events=args.fixture_events,
            max_events=args.max_events)
