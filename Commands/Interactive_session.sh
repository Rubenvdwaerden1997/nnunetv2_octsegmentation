#!/bin/bash

#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=04:00:00
#SBATCH --job-name=jupyter_lab
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/diag:/data/diag \
#SBATCH --container-image="doduo1.umcn.nl#rubenvdw/nnunetv2:1.3"
#SBATCH --exclude=dlc-meowth,dlc-arceus

# Change directory to the mounted path
cd /data/diag/rubenvdw/nnunetv2/nnUNet

# Start Jupyter Lab in the specified directory
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser