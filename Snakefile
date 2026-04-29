"""
TFlat training pipeline — Snakemake workflow.

Each run is isolated in  runs/{variant}/  where the variant string
encodes the active options so different flag combinations never clobber
each other.

Usage
-----
# Default run (truth PID labels via process.py, no dNdx feature)
snakemake --cores 1

# Include dNdx as a model feature
snakemake --cores 1 --config withdndx=True

# Add truth-matched PID probabilities as track features
snakemake --cores 1 --config withpid=True

# Use the BDT PID tool instead of truth matching (implies withpid)
snakemake --cores 1 --config withpid=True realpid=True

# Combine dNdx + BDT PID
snakemake --cores 1 --config withdndx=True withpid=True realpid=True

# Override input file and base config
snakemake --cores 1 --config input=FCCee_FT_tuples/Bs_full.root base_cfg=test_config.yaml
"""

import os
import sys
import yaml

# Python interpreter running snakemake — guaranteed to be the right env
PYTHON = sys.executable

# ── Option flags (pass via --config flag=True) ───────────────────────────────
WITH_PID  = bool(config.get("withpid",  False))
REAL_PID  = bool(config.get("realpid",  False))
WITH_DNDX = bool(config.get("withdndx", False))

INPUT    = config.get("input",    "FCCee_FT_tuples/Bd_full.root")
BASE_CFG = config.get("base_cfg", "test_config.yaml")

# ── Variant string — encodes active options for unique output paths ───────────
_tags = []
if WITH_DNDX: _tags.append("dndx")
if REAL_PID:  _tags.append("realpid")
elif WITH_PID: _tags.append("pid")
VARIANT = "_".join(_tags) if _tags else "base"

# ── Derived paths ─────────────────────────────────────────────────────────────
RUN_DIR   = f"runs/{VARIANT}"
RUN_CFG   = f"{RUN_DIR}/run_config.yaml"     # flags baked in
H5_OUT    = f"{RUN_DIR}/training.h5"
PROC_CFG  = f"{RUN_DIR}/training_cfg.yaml"   # written by process.py (includes feature counts)
CKPT      = f"{RUN_DIR}/checkpoint.model.keras"
HIST_JSON = f"{RUN_DIR}/checkpoint.history.json"
MODEL_OUT = f"{RUN_DIR}/model.keras"
HIST_PNG  = f"{RUN_DIR}/history.png"
OUT_PNG   = f"{RUN_DIR}/output.png"
LOG_DIR   = f"{RUN_DIR}/logs"


# ── Rules ─────────────────────────────────────────────────────────────────────

rule all:
    """Default target: produce both diagnostic plots."""
    input:
        HIST_PNG,
        OUT_PNG,


rule make_run_config:
    """
    Derive a run-specific config from the base config with the chosen
    setup flags applied.  process.py reads this and saves a *complete*
    version (with computed feature counts) alongside the HDF5.
    """
    input:
        BASE_CFG,
    output:
        RUN_CFG,
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f)
        cfg["setup"]["with_pid"]  = WITH_PID
        cfg["setup"]["real_pid"]  = REAL_PID
        cfg["setup"]["with_dndx"] = WITH_DNDX
        os.makedirs(RUN_DIR, exist_ok=True)
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)


rule process:
    """
    Convert ROOT tuple → flat HDF5 training file.
    Also saves a companion *_cfg.yaml with feature counts filled in
    (used by trainer and plot_output to reconstruct the model).
    """
    input:
        root = INPUT,
        cfg  = RUN_CFG,
    output:
        h5       = H5_OUT,
        proc_cfg = PROC_CFG,
    log:
        f"{LOG_DIR}/process.log",
    benchmark:
        f"{LOG_DIR}/process.tsv",
    params:
        python = PYTHON,
    shell:
        "{{ echo '=== process.py start: '$(date) &&"
        " time {params.python} process.py -i {input.root} -o {output.h5} -c {input.cfg} &&"
        " echo '=== process.py end: '$(date); }} 2>&1 | tee {log}"


rule train:
    """Build the model from the processed config and train it."""
    input:
        h5  = H5_OUT,
        cfg = PROC_CFG,
    output:
        model = MODEL_OUT,
        hist  = HIST_JSON,
    params:
        ckpt   = CKPT,
        python = PYTHON,
    log:
        f"{LOG_DIR}/train.log",
    benchmark:
        f"{LOG_DIR}/train.tsv",
    shell:
        "{{ echo '=== trainer.py start: '$(date) &&"
        " time {params.python} trainer.py"
        " -i {input.h5}"
        " -c {input.cfg}"
        " -C {params.ckpt}"
        " -m {output.model} &&"
        " echo '=== trainer.py end: '$(date); }} 2>&1 | tee {log}"


rule plot_history:
    """Plot loss / accuracy curves from the saved history JSON."""
    input:
        HIST_JSON,
    output:
        HIST_PNG,
    log:
        f"{LOG_DIR}/plot_history.log",
    benchmark:
        f"{LOG_DIR}/plot_history.tsv",
    params:
        python = PYTHON,
    shell:
        "{{ echo '=== plot_history.py start: '$(date) &&"
        " time {params.python} plot_history.py {input} -o {output} &&"
        " echo '=== plot_history.py end: '$(date); }} 2>&1 | tee {log}"


rule plot_output:
    """Plot network output distributions and print tagging metrics."""
    input:
        model = MODEL_OUT,
        data  = H5_OUT,
        cfg   = PROC_CFG,
    output:
        OUT_PNG,
    log:
        f"{LOG_DIR}/plot_output.log",
    benchmark:
        f"{LOG_DIR}/plot_output.tsv",
    params:
        python = PYTHON,
    shell:
        "{{ echo '=== plot_output.py start: '$(date) &&"
        " time {params.python} plot_output.py"
        " --model  {input.model}"
        " --data   {input.data}"
        " --config {input.cfg}"
        " -o {output} &&"
        " echo '=== plot_output.py end: '$(date); }} 2>&1 | tee {log}"
