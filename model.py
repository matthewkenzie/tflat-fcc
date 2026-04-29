#!/usr/bin/env python3

import keras


@keras.saving.register_keras_serializable(package="MyLayers")
class MyConcatenate(keras.layers.Layer):
    """Concatenate the 3D input tensors and their 2D masks along the axis=1 dimension."""

    def call(self, inputs):
        '''
        Expect the input to be list of 3D tensors
        '''
        return keras.ops.concatenate(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        '''
        Expect the mask to be list of 2D mask tensors
        '''
        if mask is None:
            return None
        return keras.ops.concatenate(mask, axis=1)

    def get_config(self):
        '''
        Keras needs this
        '''
        return super().get_config()


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer())
        mlp_layers.append(keras.layers.Dense(units, activation=activation))
        mlp_layers.append(keras.layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)


def get_tflat_model(config):
    """
    Build the TFlat transformer model.

    Feature layout of the flat input vector (must match process.py output):
        [event (num_evt_features) | tracks (num_trk × num_trk_features)
         | photons (num_photon × num_photon_features)]

    All three groups are treated as sequences (event features as a single-
    element sequence, exactly like the original ROE group) so that the
    architecture is unchanged from the original.
    """
    clip_value = config["preprocess"]["clip_value"]
    mask_value = config["preprocess"]["mask_value"]
    num_trk = config["preprocess"]["num_trk"]
    num_trk_features = config["preprocess"]["num_trk_features"]
    num_photon = config["preprocess"]["num_photon"]
    num_photon_features = config["preprocess"]["num_photon_features"]
    num_evt = config["preprocess"]["num_evt"]
    num_evt_features = config["preprocess"]["num_evt_features"]
    num_transformer_blocks = config["model"]["num_transformer_blocks"]
    num_heads = config["model"]["num_heads"]
    embedding_dims = config["model"]["embedding_dims"]
    mlp_hidden_units_factors = config["model"]["mlp_hidden_units_factors"]
    dropout_rate = config["model"]["dropout_rate"]

    # Compute start columns for event, tracks, photons
    evt_start = 0
    trk_start = num_evt * num_evt_features
    photon_start = trk_start + num_trk * num_trk_features
    n_features = photon_start + num_photon * num_photon_features

    # Create model inputs
    inputs = keras.layers.Input((n_features,))

    # Replace NaN's by a special number
    raw_features = keras.ops.nan_to_num(inputs, nan=mask_value)

    # Clip features to mitigate outliers
    raw_features = keras.ops.clip(raw_features, x_min=-clip_value, x_max=clip_value)

    # Preprocess the event features (single-element sequence)

    raw_evt_features = raw_features[:, evt_start:evt_start + num_evt * num_evt_features]
    reshaped_evt_features = keras.layers.Reshape((num_evt, num_evt_features))(raw_evt_features)
    masked_evt_features = keras.layers.Masking(mask_value=mask_value)(reshaped_evt_features)
    normed_evt_features = keras.layers.BatchNormalization()(masked_evt_features)
    encoded_evt_features = keras.layers.Dense(
        units=embedding_dims, activation=keras.activations.selu,
        name="Embedding_evt_dense_1")(normed_evt_features)
    encoded_evt_features = keras.layers.Dropout(dropout_rate, name="Embedding_evt_dropout_1")(encoded_evt_features)
    encoded_evt_features = keras.layers.BatchNormalization(name="Embedding_evt_batchnorm")(encoded_evt_features)
    encoded_evt_features = keras.layers.Dense(
        units=embedding_dims, activation=keras.activations.selu,
        name="Embedding_evt_dense_2")(encoded_evt_features)
    encoded_evt_features = keras.layers.Dropout(dropout_rate, name="Embedding_evt_dropout_2")(encoded_evt_features)

    # Preprocess the track features

    raw_trk_features = raw_features[:, trk_start:trk_start + num_trk * num_trk_features]
    reshaped_trk_features = keras.layers.Reshape((num_trk, num_trk_features))(raw_trk_features)
    masked_trk_features = keras.layers.Masking(mask_value=mask_value)(reshaped_trk_features)
    normed_trk_features = keras.layers.BatchNormalization()(masked_trk_features)
    encoded_trk_features = keras.layers.Dense(
        units=embedding_dims, activation=keras.activations.selu,
        name="Embedding_trk_dense_1")(normed_trk_features)
    encoded_trk_features = keras.layers.Dropout(dropout_rate, name="Embedding_trk_dropout_1")(encoded_trk_features)
    encoded_trk_features = keras.layers.BatchNormalization(name="Embedding_trk_batchnorm")(encoded_trk_features)
    encoded_trk_features = keras.layers.Dense(
        units=embedding_dims, activation=keras.activations.selu,
        name="Embedding_trk_dense_2")(encoded_trk_features)
    encoded_trk_features = keras.layers.Dropout(dropout_rate, name="Embedding_trk_dropout_2")(encoded_trk_features)

    # Preprocess the photon features

    raw_photon_features = raw_features[:, photon_start:photon_start + num_photon * num_photon_features]
    reshaped_photon_features = keras.layers.Reshape((num_photon, num_photon_features))(raw_photon_features)
    masked_photon_features = keras.layers.Masking(mask_value=mask_value)(reshaped_photon_features)
    normed_photon_features = keras.layers.BatchNormalization()(masked_photon_features)
    encoded_photon_features = keras.layers.Dense(
        units=embedding_dims, activation=keras.activations.selu,
        name="Embedding_photon_dense_1")(normed_photon_features)
    encoded_photon_features = keras.layers.Dropout(dropout_rate, name="Embedding_photon_dropout_1")(encoded_photon_features)
    encoded_photon_features = keras.layers.BatchNormalization(name="Embedding_photon_batchnorm")(encoded_photon_features)
    encoded_photon_features = keras.layers.Dense(
        units=embedding_dims, activation=keras.activations.selu,
        name="Embedding_photon_dense_2")(encoded_photon_features)
    encoded_photon_features = keras.layers.Dropout(dropout_rate, name="Embedding_photon_dropout_2")(encoded_photon_features)

    # Concatenate all encoded features and their masks
    encoded_features = MyConcatenate()([encoded_trk_features, encoded_photon_features, encoded_evt_features])

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_features, encoded_features)
        # Skip connection 1.
        x = keras.layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_features]
        )
        # Layer normalization 1.
        x = keras.layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = keras.layers.Dense(units=3*embedding_dims, activation='relu',
                                                name=f"feedforward_{block_idx}_dense_1")(x)
        feedforward_output = keras.layers.Dense(units=embedding_dims, name=f"feedforward_{block_idx}_dense_2")(feedforward_output)
        feedforward_output = keras.layers.Dropout(dropout_rate, name=f"feedforward_{block_idx}_dropout")(feedforward_output)
        # Skip connection 2.
        x = keras.layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_features = keras.layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Pool the "contextualized" embeddings of the features.
    features = keras.layers.GlobalAveragePooling1D()(encoded_features)

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=keras.layers.BatchNormalization,
        name="ClassifierMLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = keras.layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
