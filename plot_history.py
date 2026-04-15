#!/usr/bin/env python3
"""Plot training history (loss and accuracy vs epoch) from a saved JSON file."""

import json
import argparse
import matplotlib.pyplot as plt


def plot_history(history_path, output_path=None):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["accuracy"], label="Train")
    ax2.plot(epochs, history["val_accuracy"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("history", nargs="?", default="./ckpt/checkpoint.history.json",
                        help="Path to history JSON file")
    parser.add_argument("-o", "--output", default=None,
                        help="Save plot to file (e.g. history.png) instead of displaying")
    args = parser.parse_args()
    plot_history(args.history, args.output)
