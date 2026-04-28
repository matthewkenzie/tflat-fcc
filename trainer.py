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
    parser.add_argument("-i", "--input", default="training_data.h5",
                        help="Path to HDF5 training file")
    parser.add_argument("-c", "--config", default=None,
                        help="Path to config YAML. Default will find it based on input.")
    parser.add_argument("-C", "--checkpoint", default="./ckpt/checkpoint.model.keras",
                        help="Path to checkpoint")
    parser.add_argument("-q", "--warmstart", action=argparse.BooleanOptionalAction,
                        help="Start from checkpoint")
    parser.add_argument("-m", "--model-output", default="model.keras",
                        help="Path to save the final trained model")
    args = parser.parse_args()
    
    if not args.config:
        if os.path.exists(args.input.replace('.h5', '_cfg.yaml')):
            args.config = args.input.replace('.h5', '_cfg.yaml')
        else:
            raise RuntimeError(f"Cannot find a config file associated to {args.input_file}. Please pass --config")
    config = load_config(args.config)

    if not args.warmstart:
        if os.path.isfile(args.checkpoint):
            os.remove(args.checkpoint)

        model = get_tflat_model(config)

        cosine_decay_scheduler = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config["training"]["initial_learning_rate"],
            decay_steps=config["training"]["decay_steps"],
            alpha=config["training"]["alpha"],
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=cosine_decay_scheduler,
            weight_decay=config["training"]["weight_decay"],
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

    model.save(args.model_output)
