#!/bin/bash
#SBATCH --account=KSTARKSTAR-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=tf_training_%j.out
#SBATCH --error=tf_training_%j.err

# Exit your current python session (Ctrl+D)

# 1. Reset and Load
module purge
module load rhel8/slurm rhel8/global
module load cuda/12.1 cudnn/8.9_cuda-12.1

# 2. Map all possible library locations for TensorFlow
export CUDNN_LIB_DIR=/usr/local/Cluster-Apps/cudnn/8.9_cuda-12.1/lib64
export CUDA_LIB_DIR=/usr/local/cuda/lib64
export NVIDIA_DRIVER_DIR=/usr/local/nvidia/lib64

# 3. Update the Linker Path (The Priority Order matters)
export LD_LIBRARY_PATH=$CUDNN_LIB_DIR:$CUDA_LIB_DIR:$NVIDIA_DRIVER_DIR:$LD_LIBRARY_PATH

# 4. Critical Ampere/A100 Flags
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 5. Re-activate your environment
source /rds/project/rds-bUoO2gzww9Q/software/environments/tf-fcc-env/bin/activate

# 4. Run the training script from tflat-fcc
# Replace 'train_script.py' with the actual script name in the package
python trainer.py -i /rds/project/rds-bUoO2gzww9Q/fcc/Bd_training_pid.h5 -c config.yaml -C /rds/project/rds-bUoO2gzww9Q/fcc/Bd_training_pid.checkpoint.model.keras -m /rds/project/rds-bUoO2gzww9Q/fcc/Bd_training_pid.model.keras
