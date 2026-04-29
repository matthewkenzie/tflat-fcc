#!/usr/bin/env python3

import json
import keras
import numpy as np
import h5py


class HDF5BatchGenerator(keras.utils.PyDataset):
    """
    Generator that reads training data from a chunked HDF5 file.

    The file is read in contiguous chunks (fast for HDF5), each chunk is
    shuffled in memory, and batches are served from the chunk until it is
    exhausted, at which point the next chunk is loaded.

    Train / validation splitting is controlled via *start_idx* / *end_idx*
    (the HDF5 file stores the full, pre-shuffled dataset).
    """

    def __init__(self, h5_path, batch_size, chunk_size,
                 start_idx=0, end_idx=None, shuffle=True):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=10)
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.start_idx = start_idx

        with h5py.File(h5_path, "r") as hf:
            total = hf["X"].shape[0]
        self.end_idx = end_idx if end_idx is not None else total
        self.n_samples = self.end_idx - self.start_idx

        # Chunk management
        self.n_chunks = max(1, (self.n_samples + self.chunk_size - 1) // self.chunk_size)
        self.current_chunk_idx = 0
        self._load_chunk()
        self.current_batch = 0

    def __len__(self):
        """Total number of complete batches in the dataset."""
        return self.n_samples // self.batch_size

    def __getitem__(self, idx):
        """Return the next batch, loading a new chunk when needed."""
        if self.current_batch >= self.max_batches:
            self.current_chunk_idx = (self.current_chunk_idx + 1) % self.n_chunks
            self._load_chunk()
            self.current_batch = 0

        i0 = self.current_batch * self.batch_size
        i1 = i0 + self.batch_size
        self.current_batch += 1
        return self.X_chunk[i0:i1], self.y_chunk[i0:i1]

    def _load_chunk(self):
        """Load the next contiguous chunk from disk and optionally shuffle."""
        chunk_start = self.start_idx + self.current_chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.end_idx)

        with h5py.File(self.h5_path, "r") as hf:
            self.X_chunk = hf["X"][chunk_start:chunk_end]
            self.y_chunk = hf["y"][chunk_start:chunk_end]

        if self.shuffle:
            perm = np.random.permutation(len(self.X_chunk))
            self.X_chunk = self.X_chunk[perm]
            self.y_chunk = self.y_chunk[perm]

        self.max_batches = len(self.X_chunk) // self.batch_size


def fit(model, h5_path, config, checkpoint_filepath):
    """Train *model* on the HDF5 dataset at *h5_path*."""
    batch_size = config["training"]["batch_size"]
    chunk_size = config["training"]["chunk_size"]
    train_frac = config["training"].get("train_valid_fraction", 0.9)

    with h5py.File(h5_path, "r") as hf:
        n_total = hf["X"].shape[0]
    split = int(n_total * train_frac)

    train_ds = HDF5BatchGenerator(
        h5_path, batch_size, chunk_size, start_idx=0, end_idx=split, shuffle=True
    )
    val_ds = HDF5BatchGenerator(
        h5_path, batch_size, chunk_size, start_idx=split, end_idx=n_total, shuffle=False
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config["training"]["patience"],
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, monitor="val_loss",
            mode="min", save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Save history so it can be plotted later
    history_path = checkpoint_filepath.replace(".model.keras", ".history.json")
    with open(history_path, "w") as f:
        # Convert numpy types to plain Python for JSON serialisation
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"Training history saved to {history_path}")

    return history
