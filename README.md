# TFlaT - FCC

This README describes the training process of the transformer based flavortagger TFlaT with FCC simulation.
This is inspired by the Belle-II flavour tagger `TFlaT` from https://github.com/BenjaminSchwenker/tflat/tree/main.
The provided scripts cover the steps required to get from IDEA simulation tuples to the training. 

The input tuples (produced by Ella Wood) can be found here:

`/eos/user/e/ewood/FCCee_FT_tuples`

---

## Setup the software

Create a conda environment by executing lines below:

```
git clone https://github.com/BenjaminSchwenker/tflat.git
cd tflat

conda create -n tflat  python=3.11.9
conda activate tflat

pip install 'tensorflow[and-cuda]'
pip install pandas
pip install pyarrow
pip install PyYAML
pip install uproot
pip install awkward
pip install matplotlib
pip install xgboost
pip install scikit-learn
```

Confirm that tensorflow is seeing your GPU by typing:

```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

In case the GPU device is not found, your system may be missing GPU drivers. Please
follow the instructions on this page: https://www.tensorflow.org/install/pip

---

## Other packages

If you want to run the PID resampling you also need the `get_pid` tool from here:
https://github.com/anjabeck/fccee-tracker-pid/tree/main 

Which will be searched for in the parent directory.

---

## Hardware Requirements

*This section is inherited from the Belle-II model. It will need updating for the FCC attempts.*

The training process requires a CUDA capable GPU.\
Your GPU should have 8GB VRAM and your PC should have 16GB RAM.\
The time needed to complete a training depends on the specific GPU. For a NVIDIA A100 GPU the expected time to completion with 10M training samples is ~1 days.\

---

## Step-by-Step Guide

1. **Data preparation**

 - Prepare the training file (`hd5` format is used because it can read very efficiently from chunks). A flat `.root` version will also be prepared. Run with a few different settings:

   ```bash
   mkdir training_files
   
   # Run Bd jobs
   python process.py -i Bd_full.root -o training_files/Bd_training.h5
   python process.py -i Bd_full.root -o training_files/Bd_training.h5 -s # adds dN/dx var to feature set
   python process.py -i Bd_full.root -o training_files/Bd_training.h5 -p # uses PID tool to get realistic PID vars
   
   # Run Bs jobs
   python process.py -i Bs_full.root -o training_files/Bs_training.h5
   python process.py -i Bs_full.root -o training_files/Bs_training.h5 -s # adds dN/dx var to feature set
   python process.py -i Bs_full.root -o training_files/Bs_training.h5 -p # uses PID tool to get realistic PID vars
   ```

 - There is some metadata saved in the output training file as well as the training arrays.

2. **Training**

 - Launch the training with the `trainer.py` script:

    ```bash
    python trainer.py 
    ```

 - If the training crashes at any point it can be restarted from the latest checkpoint with:

    ```bash
    python trainer.py --warmstart
    ```

 - Once the training is done a keras weightfile is produced called `model.keras`

 - The training history will also be saved with the checkpoint

3. **Plotting and performance metrics**

 - You can plot the training history using

    ```bash
    python plot_history.py ckpt/checkpoint.history.json -o history.png
    ```

 - You can plot the output network and get flavour tagging performance metrics using

    ```bash
    python plot_output.py -o output.png
    ```
    
    This will also save the performance metrics in a file called `output.json`
