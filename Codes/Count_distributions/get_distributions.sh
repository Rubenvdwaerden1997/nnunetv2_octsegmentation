#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH -o _slurm_output_count_%j.txt
#SBATCH -e _slurm_error_count_%j.txt
#SBATCH --qos=high
#SBATCH --exclude=dlc-meowth,dlc-arceus,dlc-articuno

python3 -u /data/diag/rubenvdw/nnunetv2/nnUNet/Codes/Count_distributions/get_distributions.py \
    --data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset501/labelsTs \
    --output_filename /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset501/Metrics/Counts_labels \
    --data_info /data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_split_extraframes_13062024 \
    --num_classes 15 \
    --lipid_cap_area_manual 0 \
    --task_name Dataset501

    #--data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset905_SegmentOCT3d3/labelsTs \
    #--data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files \
    #--data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files_postprocessing \
