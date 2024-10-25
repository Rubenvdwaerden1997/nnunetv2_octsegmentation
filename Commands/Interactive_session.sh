#!/bin/bash

#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=04:00:00
#SBATCH --job-name=jupyter_lab
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/diag:/data/diag \
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"

# Change directory to the mounted path
cd /data/diag/rubenvdw/nnunetv2/nnUNet

# Start Jupyter Lab in the specified directory
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser