#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH --qos=high
#SBATCH -o _slurm_output_%j.txt
#SBATCH -e _slurm_error_%j.txt
#SBATCH --exclude=dlc-arceus,dlc-meowth,dlc-nidoking,dlc-articuno,dlc-moltres

cd /data/diag/rubenvdw/nnunetv2/nnUNet/Codes/Dataset_conversion

# Print the hostname (GPU node) the job is running on
echo "Running on GPU node: $(hostname)"

pip3 install --upgrade einops

# Run the script
python3 -u Dataset_conversion_imgs_segs.py \
    --data_input '/data/diag/rubenvdw/Dataset/DICOMS_15classes' '/data/diag/rubenvdw/Dataset/DICOMS_Orange' \
    --data_label '/data/diag/rubenvdw/Dataset/SEGMENTATIONS_15classes_25102024' \
    --outfolder '/data/diag/rubenvdw/nnunetv2/nnUnet/nnunetv2/Data/nnUNet_raw' \
    --task Dataset601 \
    --k 3 \
    --radius 352 \
    --splitfile '/data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_split_extraframes_25102024.xlsx' \