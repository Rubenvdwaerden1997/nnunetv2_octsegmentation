#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH -o _slurm_output_count_%j.txt
#SBATCH -e _slurm_error_count_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-meowth,dlc-arceus,dlc-articuno

python3 -u /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Count_distributions/get_distributions_multiprocessing.py \
    --data_path '/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset601_TS3D3/labelsTr' \
    --output_filename /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Metrics/Counts_trainingset \
    --data_info /data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_newsplit_29102024 \
    --num_classes 14 \
    --lipid_cap_area_manual 0 \
    --task_name Dataset601_TS3D3

    #--data_path '/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset701_TS3D3/labelsTs' '/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset701_TS3D3/labelsTr' \
    #--data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset905_SegmentOCT3d3/labelsTs \
    #--data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files \
    #--data_path /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files_postprocessing \