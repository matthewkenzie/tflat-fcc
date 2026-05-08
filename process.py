import os
import sys
import importlib.util
import uproot
import awkward as ak
import numpy as np
import h5py
import argparse
import tensorflow as tf
from tqdm import tqdm
np.random.seed(210187)

import utils

# Path to the fccee-tracker-pid package (sibling directory)
# PID_PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fccee-tracker-pid")

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

def _load_pid_tools(pid_tool_loc):
    """Import helper functions from fccee-tracker-pid without triggering its CLI entry point.

    get_pid.py runs argparse + assertions at module level, so we load it via
    importlib and catch the SystemExit/AssertionError that fires before the
    tree-reading code, by which point the three functions are already defined.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pid_tool_loc)
    spec = importlib.util.spec_from_file_location(
        "pid_tools", os.path.join(path, "get_pid.py"))
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


def get_pid(tracks, absid, pid_tool_loc="fccee-tracker-pid", detector="IDEA", tof_val=-1.0, dndx_val=0.8):
    """Apply BDT-based PID from fccee-tracker-pid for pi/K/p; truth-match e/mu."""
    read_classifiers, make_prediction, get_features = _load_pid_tools(pid_tool_loc)
    usededx = "nodedx"

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pid_tool_loc)

    # Model paths inside the package are relative — change CWD temporarily
    orig_dir = os.getcwd()
    os.chdir(path)
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


def process(input_file, output_file, cfg):
    """Read raw ROOT, process, pad, flatten, and save to HDF5."""
    
    dndx = cfg["setup"]["with_dndx"]
    pid = cfg["setup"]["with_pid"]
    real_pid = cfg["setup"]["real_pid"]

    max_events = cfg["preprocess"]["max_events"]
    max_ntracks = cfg["preprocess"]["num_trk"]
    max_nphotons = cfg["preprocess"]["num_photon"]

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
    cfg["preprocess"]["num_photon_features"] = len(photons.fields)

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

    if pid:
        if real_pid:
            tracks = get_pid(tracks, absid, cfg["setup"]["pid_package_loc"])
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
    cfg["preprocess"]["num_evt_features"] = len(GLOB_VARS)

    # ── Target  ─────────────────────────────────────────────────────────────
    target = tree.arrays(["MC_B_qTag"], **kw)
    target_flat = target["MC_B_qTag"].to_numpy().astype(np.int32)
    # check no untagged
    assert( len(target_flat[target_flat==0])==0 )
    # shift to [0, 1]
    target_flat[target_flat==-1] = 0

    # ── Pad and flatten variable-length arrays ────────────────────────────────────
    # Derive from what's actually in the tracks record so pid=False runs
    # don't try to flatten track_prob_* that were never added.
    track_field_set = set(tracks.fields)
    trk_feats = [f for f in TRACK_FEATURES
                 if f in track_field_set and (dndx or f != "track_dndx")]
    cfg["preprocess"]["num_trk_features"] = len(trk_feats)
    track_flat  = pad_and_flatten(tracks,  trk_feats,  max_ntracks)
    photon_flat = pad_and_flatten(photons, PHOTON_FEATURES, max_nphotons)

    # ── Concatenate: event | tracks | photons ───────────────────────────────
    X = np.concatenate([event_flat, track_flat, photon_flat], axis=1)
    y = target_flat

    # ── Shuffle ─────────────────────────────────────────────────────────────
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # ── Build ordered feature name list ───────────────────────────────────────
    feature_names = list(EVENT_FEATURES)
    for i in range(max_ntracks):
        feature_names += [f"{f}_{i}" for f in trk_feats]
    for i in range(max_nphotons):
        feature_names += [f"{f}_{i}" for f in PHOTON_FEATURES]

    # ── Save to HDF5 ────────────────────────────────────────────────────────
    chunk_rows = min(cfg["training"]["chunk_size"], len(X))
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
        hf.attrs["max_ntracks"]       = max_ntracks
        hf.attrs["max_nphotons"]      = max_nphotons
        hf.attrs["event_features"]    = EVENT_FEATURES
        hf.attrs["track_features"]    = trk_feats 
        hf.attrs["photon_features"]   = PHOTON_FEATURES
        hf.attrs["feature_names"]     = feature_names
    
    # ── Save to root ────────────────────────────────────────────────────────
    branches = {name: X[:, i] for i, name in enumerate(feature_names)}
    branches["qTag"] = y
    with uproot.recreate(output_file.replace(".h5",".root")) as outf:
        outf["training"] = branches

    # ── Save to tfrecord  ────────────────────────────────────────────────────────
    n_samples = X.shape[0]

    # shuffled index array
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # determine split point
    split_idx = int(n_samples * cfg["training"]["train_valid_fraction"])
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    def write_records(target_path, index_list):
        print(f"Writing to {target_path}...")
        with tf.io.TFRecordWriter(target_path) as writer:
            for i in tqdm(index_list):
                # Fetch single row from H5 (H5 handles this efficiently via indexing)
                features = X[i].astype('float32')
                label = y[i].astype('float32')
                
                # Create the Example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature={
                    'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                }))
                writer.write(example.SerializeToString())

    # 3. Execute the writes
    train_path = output_file.replace(".h5", ".tfrecord")
    if 'training' in train_path:
        valid_path = train_path.replace("training", "validation")
    elif 'train' in train_path:
        valid_path = train_path.replace("train", "valid")
    else:
        valid_path = train_path.replace(".tfrecord", "_valid.tfrecord")

    write_records(train_path, train_indices)
    write_records(valid_path, valid_indices)

    print(f'Saved {len(X)} events × {X.shape[1]} features to {output_file}, {output_file.replace(".h5", ".root")}, {train_path} and {valid_path}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", default="Bd_full.root", help="Input tuple")
    parser.add_argument("-o", "--output", default="Bd_training.h5", help="Output training file")
    parser.add_argument("-c", "--config-file", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = utils.load_config(args.config_file)

    process(args.input, args.output, cfg)
    
    utils.save_config(args.output.replace(".h5", ".config.yaml"), cfg)

