#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=168:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/diag:/data/diag,/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data:/home/user/Data
#SBATCH --container-image="doduo1.umcn.nl#rubenvdw/nnunetv2:1.3"
#SBATCH --qos=high
#SBATCH -o ./Slurm/_slurm_output_plan_%j.txt
#SBATCH -e ./Slurm/_slurm_error_plan_%j.txt

echo "Running on GPU node: $(hostname)"

# python3 -u nnunetv2/environmental_paths.py 

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

python3 -u nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py \
    -d 501 \
    -c 2d \