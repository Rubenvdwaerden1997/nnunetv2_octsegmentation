#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/diag:/data/diag,/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data:/home/user/Data
#SBATCH --container-image="doduo1.umcn.nl#rubenvdw/nnunetv2:1.3"
#SBATCH --qos=high
#SBATCH -o ./Slurm/_slurm_output_pipeline_%j.txt
#SBATCH -e ./Slurm/_slurm_error_pipeline_%j.txt
#SBATCH --exclude=dlc-meowth,dlc-arceus,dlc-articuno

pip3 install --upgrade natsort

# Base directory
BASE_DIR="/home/user/Data"

# Scripts and their arguments
SCRIPTS=(
        "/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Dataset_conversion/Dataset_conversion_imgs_segs.py,--data_input,/data/diag/rubenvdw/Dataset/DICOMS_15classes,--k,3,--radius,352,--preprocessing"
        "/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/nnunetv2/inference/predict_from_raw_data.py,-i,/data/diag/rubenvdw/Dataset/tempfolder0,-o,/data/diag/rubenvdw/Dataset/tempfolder1,-d,601,-c,2d,--save_probabilities"
        "/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils/Concatenate_predictions.py,--folder_path,/data/diag/rubenvdw/Dataset/tempfolder1,--output_path,/data/diag/rubenvdw/Dataset/Output_DICOMS_15classes,--file_extension,.nii,--concatenate_axis,0"
        )

# Run the Python script with arguments
python3 -u /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Pipeline/Pipeline_onepullback_v2.py  \
        --base_dir "$BASE_DIR" \
        --scripts "${SCRIPTS[@]}" \
        --input_folder "/data/diag/rubenvdw/Dataset/DICOMS_15classes" \