#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH --qos=high
#SBATCH -o _slurm_output_%j.txt
#SBATCH -e _slurm_error_%j.txt
#SBATCH --exclude=dlc-meowth,dlc-nidoking,dlc-articuno
#SBATCH --nodelist=dlc-arceus

cd /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Dataset_conversion

# Print the hostname (GPU node) the job is running on
echo "Running on GPU node: $(hostname)"

pip3 install --upgrade einops

# Run the script
python3 -u Dataset_conversion_imgs_segs.py \
    --data_input '/data/diag/rubenvdw/Dataset/DICOMS_15classes' \
    --k 3 \
    --radius 352 \
    --preprocessing \
    --data_label '/data/diag/rubenvdw/Dataset/SEGMENTATIONS_15classes_25102024' \
    --outfolder '/data/diag/rubenvdw/nnunetv2/nnUnet/nnunetv2/Data/nnUNet_raw' \
    --task Dataset711_TS3D3 \
    --trainmode \
    --splitfile '/data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_newsplit_29102024.xlsx' \