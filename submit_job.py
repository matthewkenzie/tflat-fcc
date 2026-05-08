import os
import sys
import subprocess

import utils

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
args = parser.parse_args()

config = utils.load_config(args.config)

assert(config["setup"]["variant"] in ["Bd", "Bs"])

inpath = config["setup"]["inpath"]
outpath = config["setup"]["outpath"]
variant = config["setup"]["variant"]

subdir = "base"
if config["setup"]["with_dndx"]:
    subdir = "dndx"
elif config["setup"]["with_pid"]:
    subdir = "pid"
if config["setup"]["real_pid"]:
    subdir = "realpid"

RAW_FILE = f"{inpath}/{variant}_full.root" 
OUT_PATH = f"{outpath}/{variant}/{subdir}"
CFG_FILE = args.config
TRAIN_FILE = f"{OUT_PATH}/training.h5"
TRAIN_TFFILE = f"{OUT_PATH}/training.tfrecord"
TRAIN_CFG = f"{OUT_PATH}/training.config.yaml"
MODEL_FILE = f"{OUT_PATH}/model.keras"
CKPT_FILE = f"{OUT_PATH}/checkpoint.model.keras"
HIST_FILE = f"{OUT_PATH}/checkpoint.history.json"
OUTPUT_PLOT = f"{OUT_PATH}/output.png"
OUTPUT_JSON = f"{OUT_PATH}/output.json"
HISTORY_PLOT = f"{OUT_PATH}/history.png"

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH, exist_ok=True)

# 1.) Preprocess
rerun = True

if os.path.exists(TRAIN_TFFILE):
    inp = input(f"Training file {TRAIN_TFFILE} already exists. Do you want to overwrite it?\n")
    if inp in ["y","Y","yes","Yes"]:
        rerun = True
    else:
        rerun = False

if rerun:
    cmds = ["python", "process.py", "-i", RAW_FILE, "-o", TRAIN_FILE, "-c", CFG_FILE]
    print("RUN PREPROCESSING....")
    print("    ", " ".join(cmds))
    subprocess.run(cmds)


# 2.) Training
retrain = True

if os.path.exists(MODEL_FILE):
    inp = input(f"Model file {MODEL_FILE} already exists. Do you want to retrain it?\n")
    if inp in ["y","Y","yes","Yes"]:
        retrain = True
    else:
        retrain = False

if retrain:
    # make job script
    with open("job_script.sh") as f:
        lines = f.readlines()

    with open(f"{OUT_PATH}/job_script.sh", "w") as f:
        for line in lines[:-1]:
            if "--output" in line:
                line = f"#SBATCH --output={OUT_PATH}/training.stdout\n"
            if "--error" in line:
                line = f"#SBATCH --error={OUT_PATH}/training.stderr\n"
            
            f.write(line)

        cmd = f"python trainer.py -i {TRAIN_TFFILE} -c {TRAIN_CFG} -C {CKPT_FILE} -m {MODEL_FILE}"
        f.write(cmd+"\n")

    inp = input(f"Job script made at {OUT_PATH}/job_script.sh. Would you like to submit it?\n")

    if inp in ["y","Y","yes","Yes"]:
        cmds = ["sbatch", f"{OUT_PATH}/job_script.sh"]
        print("SUBMITTING GPU JOB...")
        subprocess.run(cmds)
        sys.exit("Check progress on training and rerun this script to make output plots when it finishes")

# 3.) Make plots
replot = True

if os.path.exists(HISTORY_PLOT) and os.path.exists(OUTPUT_PLOT):
    inp = input(f"Output plots {HISTORY_PLOT} and {OUTPUT_PLOT} already exist. Do you want to replot?\n")
    if inp in ["y","Y","yes","Yes"]:
        replot = True
    else:
        replot = False

if replot:
    print("MAKING PLOTS...")
    subprocess.run(["python", "plot_history.py", HIST_FILE, "-o", HISTORY_PLOT])
    subprocess.run(["python", "plot_output.py", "-m", MODEL_FILE, "-d", TRAIN_TFFILE, "-c", TRAIN_CFG, "-o", OUTPUT_PLOT])
