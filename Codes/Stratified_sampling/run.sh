#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH --qos=high
#SBATCH -o _slurm_output_%j.txt
#SBATCH -e _slurm_error_%j.txt
#SBATCH --exclude=

cd /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Stratified_sampling

# Print the hostname (GPU node) the job is running on
echo "Running on GPU node: $(hostname)"

pip3 install --upgrade einops

# Run the script
python3 -u Stratified_split_traintest.py \