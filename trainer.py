#!/usr/bin/env python3

import os
import argparse


if __name__ == "__main__":

    import keras
    from fitter import fit
    from utils import load_config
    from model import get_tflat_model

    parser = argparse.ArgumentParser(
        description="Train TFlat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", default="training_data.h5",
                        help="Path to HDF5 training file")
    parser.add_argument("--configFile", default="config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", default="./ckpt/checkpoint.model.keras",
                        help="Path to checkpoint")
    parser.add_argument("--warmstart", action=argparse.BooleanOptionalAction,
                        help="Start from checkpoint")
    args = parser.parse_args()

    config = load_config(args.configFile)
    parameters = config["parameters"]

    if not args.warmstart:
        if os.path.isfile(args.checkpoint):
            os.remove(args.checkpoint)

        model = get_tflat_model(parameters=parameters)

        cosine_decay_scheduler = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config["initial_learning_rate"],
            decay_steps=config["decay_steps"],
            alpha=config["alpha"],
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=cosine_decay_scheduler,
            weight_decay=config["weight_decay"],
        )
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.binary_crossentropy,
            metrics=["accuracy", keras.metrics.AUC(), keras.metrics.MeanSquaredError()],
        )
    else:
        model = keras.models.load_model(args.checkpoint)

    model.summary()

    history = fit(model, args.input, config, args.checkpoint)

    model.save("model.keras")
