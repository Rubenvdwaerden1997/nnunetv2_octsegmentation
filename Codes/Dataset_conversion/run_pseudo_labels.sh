#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH --qos=low
#SBATCH -o _slurm_output_%j.txt
#SBATCH -e _slurm_error_%j.txt
#SBATCH --exclude=dlc-arceus,dlc-meowth,dlc-nidoking

cd /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Dataset_conversion

# Print the hostname (GPU node) the job is running on
echo "Running on GPU node: $(hostname)"

pip3 install --upgrade einops

# Run the script
python3 -u Dataset_conversion_imgs_pseudolabels.py \
    --data_input '/data/diag/rubenvdw/Dataset/DICOMS_15classes' '/data/diag/rubenvdw/Dataset/DICOMS_Orange' \
    --outfolder '/data/diag/rubenvdw/nnunetv2/nnUnet/nnunetv2/Data/nnUNet_raw' \
    --task Pseudo_labels_complete_trainingset \
    --k 3 \
    --radius 352 \
    --splitfile '/data/diag/rubenvdw/Info_files_Dataset_split/15_classes_dataset_newsplit_29102024.xlsx' \