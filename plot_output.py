#!/usr/bin/env python3
"""
Plot the network output distribution for training and validation samples.

Training: filled histograms
Validation: data points with Poisson errors
Colours: red = target 0, blue = target 1
"""

import argparse
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt


def plot_output(model_path, data_path, config_path, output_path=None, n_bins=30):
    from utils import load_config
    import keras
    import model as _  # noqa: F401 — registers MyConcatenate for deserialization

    config = load_config(config_path)
    train_frac = config["training"].get("train_valid_fraction", 0.9)

    # Load data and compute train/val split (must match fitter.py)
    with h5py.File(data_path, "r") as hf:
        X = hf["X"][:]
        y = hf["y"][:]
    split = int(len(X) * train_frac)

    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Get predictions
    model = keras.models.load_model(model_path)
    pred_train = model.predict(X_train, verbose=0).ravel()
    pred_val = model.predict(X_val, verbose=0).ravel()

    # Compute flavour tagging metrics
    #   tag decision: pred > 0.5 → flavour 1, else → flavour 0
    #   ω  = mistag rate (fraction of wrong tags)
    #   D  = dilution = 1 − 2ω
    #   P  = tagging power = ε_tag × D²  (ε_tag = 1 here)
    def tagging_power(pred, truth):
        N = len(truth)
        tag = (pred > 0.5).astype(int)
        omega = np.mean(tag != truth)
        D = 1 - 2 * omega
        P = D**2
        # Binomial uncertainties
        sigma_omega = np.sqrt(omega * (1 - omega) / N)
        sigma_D = 2 * sigma_omega
        sigma_P = 4 * abs(D) * sigma_omega
        return omega, D, P, sigma_omega, sigma_D, sigma_P

    omega_tr, D_tr, P_tr, s_omega_tr, s_D_tr, s_P_tr = tagging_power(pred_train, y_train)
    omega_val, D_val, P_val, s_omega_val, s_D_val, s_P_val = tagging_power(pred_val, y_val)

    # Collect metrics into a dict for saving / printing
    metrics = {
        "train": {
            "N": len(y_train),
            "w": omega_tr, "sigma_w": s_omega_tr,
            "D": D_tr, "sigma_D": s_D_tr,
            "P": P_tr, "sigma_P": s_P_tr,
        },
        "val": {
            "N": len(y_val),
            "w": omega_val, "sigma_w": s_omega_val,
            "D": D_val, "sigma_D": s_D_val,
            "P": P_val, "sigma_P": s_P_val,
        },
    }

    # Print to terminal
    for sample in ("train", "val"):
        m = metrics[sample]
        print(f"{sample:>5s} (N={m['N']:>5d}):  "
              f"w = {m['w']:.4f} +/- {m['sigma_w']:.4f}   "
              f"D = {m['D']:.4f} +/- {m['sigma_D']:.4f}   "
              f"P = {m['P']:.5f} +/- {m['sigma_P']:.5f}")

    # Save to JSON
    metrics_path = (output_path.rsplit(".", 1)[0] + ".json") if output_path else "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Common binning
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    fig, (ax, ax_pull) = plt.subplots(
        2, 1, figsize=(8, 7), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # --- Training: filled histograms ---
    # Keep the raw counts so we can compute pulls against validation
    train_densities = {}
    train_errs = {}
    for target, colour in [(0, "red"), (1, "blue")]:
        counts_tr, _ = np.histogram(pred_train[y_train == target], bins=bins)
        n_tr = (y_train == target).sum()
        train_densities[target] = counts_tr / (n_tr * bin_width) if n_tr > 0 else counts_tr * 0.0
        train_errs[target] = np.sqrt(counts_tr) / (n_tr * bin_width) if n_tr > 0 else counts_tr * 0.0

    ax.hist(pred_train[y_train == 0], bins=bins, density=True,
            histtype="stepfilled", alpha=0.35, color="red", edgecolor="red",
            label="Train (target=0)")
    ax.hist(pred_train[y_train == 1], bins=bins, density=True,
            histtype="stepfilled", alpha=0.35, color="blue", edgecolor="blue",
            label="Train (target=1)")

    # --- Validation: data points with Poisson errors ---
    val_densities = {}
    val_errs = {}
    for target, colour, marker in [(0, "red", "v"), (1, "blue", "^")]:
        counts, _ = np.histogram(pred_val[y_val == target], bins=bins)
        n_class = (y_val == target).sum()
        density = counts / (n_class * bin_width) if n_class > 0 else counts * 0.0
        err = np.sqrt(counts) / (n_class * bin_width) if n_class > 0 else counts * 0.0
        val_densities[target] = density
        val_errs[target] = err
        ax.errorbar(bin_centres, density, yerr=err, fmt=marker, color=colour,
                     markersize=5, linewidth=1.2, capsize=2,
                     label=f"Val (target={target})")

    ax.set_ylabel("Normalised density")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, fontsize=9,
              borderaxespad=0, frameon=False)

    # Print tagging power on the plot
    text = (
        f"Train: $\\omega$={omega_tr:.3f}±{s_omega_tr:.3f}"
        f"  D={D_tr:.3f}±{s_D_tr:.3f}"
        f"  $\\varepsilon D^2$={P_tr:.4f}±{s_P_tr:.4f}\n"
        f"Val:   $\\omega$={omega_val:.3f}±{s_omega_val:.3f}"
        f"  D={D_val:.3f}±{s_D_val:.3f}"
        f"  $\\varepsilon D^2$={P_val:.4f}±{s_P_val:.4f}"
    )
    ax.text(0.50, 0.97, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            family="monospace")

    # --- Pull plot: (val - train) / quadrature error ---
    for target, colour, marker in [(0, "red", "v"), (1, "blue", "^")]:
        quad_err = np.sqrt(train_errs[target]**2 + val_errs[target]**2)
        diff = val_densities[target] - train_densities[target]
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = np.where(quad_err > 0, diff / quad_err, 0.0)
        pull_errs = np.ones_like(pull)
        ax_pull.errorbar(bin_centres, pull, pull_errs, marker=marker, color=colour,
                     markersize=5, linewidth=1.2, linestyle='none', capsize=2, label=f"target={target}")

    ax_pull.axhline(0, color="grey", linewidth=0.8)
    ax_pull.set_xlabel("Network output")
    ax_pull.set_ylabel("Pull")
    ax_pull.set_ylim(-4, 4)

    fig.align_ylabels([ax, ax_pull])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot network output distributions")
    parser.add_argument("--model", default="model.keras",
                        help="Path to saved Keras model")
    parser.add_argument("--data", default="training_data.h5",
                        help="Path to HDF5 data file")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of histogram bins")
    parser.add_argument("-o", "--output", default=None,
                        help="Save plot to file instead of displaying")
    args = parser.parse_args()
    plot_output(args.model, args.data, args.config, args.output, args.bins)
