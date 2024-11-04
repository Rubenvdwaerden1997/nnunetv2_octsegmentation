#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH -o _slurm_output_pdf_%j.txt
#SBATCH -e _slurm_error_pdf_%j.txt
#SBATCH --qos=high

python3 -u /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Predictions_PDF/create_imgs_pdf.py \
    --orig /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset501/labelsTs \
    --preds /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset501/Predicted_files_postprocessing \
    --data_info /data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_split_extraframes_13062024.xlsx \
    --output_folder /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset501/Metrics \
    --dicom_folder /data/diag/rubenvdw/Dataset/DICOMS_15classes \
    --pdf_name Dataset501_postprocessing 