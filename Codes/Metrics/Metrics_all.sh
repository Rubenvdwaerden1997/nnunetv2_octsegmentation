#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH -o _slurm_output_metrics_%j.txt
#SBATCH -e _slurm_error_metrics_%j.txt
#SBATCH --qos=high
#SBATCH --exclude=dlc-meowth,dlc-arceus,dlc-articuno

python3 -u /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Metrics/Metrics_all.py \
    --orig_folder /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset601_TS3D3/labelsTs \
    --preds_folder /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Predicted_files_postprocessing \
    --data_info /data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_newsplit_29102024.xlsx \
    --filename Dataset601_TS3D3_postprocessing_v2 \
    --num_classes 14 \
    --output_folder /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Metrics \
    --counts_testset /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Metrics/Counts_labels.xlsx \
    --counts_predictiontestset /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset601_TS3D3/Metrics/Counts_predictions_postprocessing.xlsx