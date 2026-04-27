#!/usr/bin/env python3
"""
Create a small HDF5 fixture by slicing the first N events from a large training file.

Usage:
    python make_fixture.py -i training_data.h5 -n 512 -o test_fixture.h5
"""

import argparse
import h5py
import numpy as np


def make_fixture(input_path, output_path, n_events):
    with h5py.File(input_path, "r") as src:
        total = src["X"].shape[0]
        n = min(n_events, total)
        print(f"Sampling {n} / {total} events from {input_path}")

        X = src["X"][:n]
        y = src["y"][:n]

        with h5py.File(output_path, "w", track_order=True) as dst:
            chunk_rows = min(256, n)
            dst.create_dataset("X", data=X,
                               chunks=(chunk_rows, X.shape[1]),
                               compression="gzip", compression_opts=4)
            dst.create_dataset("y", data=y,
                               chunks=(chunk_rows,),
                               compression="gzip", compression_opts=4)
            # Copy all metadata attributes
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["n_events"] = n

    print(f"Saved {n} events × {X.shape[1]} features → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample a training HDF5 for unit-test fixtures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",  required=True, help="Source HDF5 training file")
    parser.add_argument("-o", "--output", default="test_fixture.h5", help="Output fixture file")
    parser.add_argument("-n", "--n-events", type=int, default=512,
                        help="Number of events to extract")
    args = parser.parse_args()
    make_fixture(args.input, args.output, args.n_events)
