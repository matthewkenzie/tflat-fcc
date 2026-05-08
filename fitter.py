#!/usr/bin/env python3

import json
import keras
import numpy as np
import h5py
import tensorflow as tf

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

def get_tfrecord_dataset(file_pattern, batch_size, n_features, is_training=True):
    """
    Creates a high-performance tf.data pipeline from TFRecord files.
    """

    # 1. Define the schema (must match what you wrote to the TFRecord)
    feature_description = {
        'features': tf.io.FixedLenFeature([n_features], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['features'], parsed['label']

    # 2. Load the dataset
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))

    if is_training:
        # Buffer size for shuffling (e.g., 10,000 samples)
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # 3. The "Magic": Prefetch lets the CPU prepare Batch N+1 while GPU runs Batch N
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def load_tfrecord_to_memory(file_pattern, batch_size, n_features, is_training=True):
    ds = get_tfrecord_dataset(file_pattern, batch_size, n_features, is_training=True)

    xs = []
    ys = []
    
    print(f"Loading TFRecord into memory")
    for batch_x, batch_y in ds.as_numpy_iterator():
        xs.append(batch_x)
        ys.append(batch_y)

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    return X, y

def fit(model, h5_path, config, checkpoint_filepath):
    """Train *model* on the HDF5 dataset at *h5_path*."""
    batch_size = config["training"]["batch_size"]
    chunk_size = config["training"]["chunk_size"]
    train_frac = config["training"]["train_valid_fraction"]
    n_features = config["preprocess"]["num_trk"] * config["preprocess"]["num_trk_features"] + config["preprocess"]["num_photon"] * config["preprocess"]["num_photon_features"] + config["preprocess"]["num_evt_features"]

    # Create the high-performance datasets
    train_path = h5_path
    if 'training' in train_path:
        valid_path = train_path.replace("training", "validation")
    elif 'train' in train_path:
        valid_path = train_path.replace("train", "valid")
    else:
        valid_path = train_path.replace(".tfrecord", "_valid.tfrecord")

    train_ds = get_tfrecord_dataset(train_path, batch_size, n_features, is_training=True)
    val_ds = get_tfrecord_dataset(valid_path, batch_size, n_features, is_training=False)
    
    #train_ds = HDF5BatchGenerator(h5_path, batch_size, chunk_size, start_idx=0, end_idx=split, shuffle=True)
    #val_ds = HDF5BatchGenerator(h5_path, batch_size, chunk_size, start_idx=split, end_idx=n_total, shuffle=False)

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
        # steps_per_epoch=len(train_ds),
        # validation_steps=len(val_ds),
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
