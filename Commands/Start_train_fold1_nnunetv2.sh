#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/diag:/data/diag,/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data:/home/user/Data
#SBATCH --container-image="doduo1.umcn.nl#rubenvdw/nnunetv2:1.3"
#SBATCH --qos=low
#SBATCH -o ./Slurm/_slurm_output_train1_%j.txt
#SBATCH -e ./Slurm/_slurm_error_train1_%j.txt
#SBATCH --exclude=dlc-meowth,dlc-arceus

echo "Running on GPU node: $(hostname)"

# Base directory
BASE_DIR="/home/user/Data"

# Set environment variables
export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${BASE_DIR}/nnUNet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

# Optional: Verify the variables are set correctly
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

python3 -u nnunetv2/run/run_training.py \
    601 \
    2d \
    1 \
    --c \