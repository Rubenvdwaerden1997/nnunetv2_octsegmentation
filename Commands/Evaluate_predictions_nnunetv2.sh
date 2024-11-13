#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=168:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/diag:/data/diag,/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data:/home/user/Data
#SBATCH --container-image="doduo1.umcn.nl#rubenvdw/nnunetv2:1.3"
#SBATCH --qos=high
#SBATCH -o ./Slurm/_slurm_output_evaluate_%j.txt
#SBATCH -e ./Slurm/_slurm_error_evaluate_%j.txt
#SBATCH --exclude=dlc-meowth,dlc-arceus,dlc-articuno

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

python3 -u nnunetv2/evaluation/evaluate_predictions.py \
    /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset601_TS3D3/labelsTs \
    /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Predicted_files_postprocessing \
    -djfile /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Predicted_files/dataset.json \
    -pfile /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Predicted_files/plans.json