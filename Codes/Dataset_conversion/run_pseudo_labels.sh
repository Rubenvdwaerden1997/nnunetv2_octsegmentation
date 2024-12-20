#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH --qos=high
#SBATCH -o _slurm_output_pseudo_%j.txt
#SBATCH -e _slurm_error_pseudo_%j.txt
#SBATCH --exclude=dlc-meowth,dlc-nidoking,dlc-articuno,dlc-electabuzz,dlc-mewtwo

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
    --output_folder Pseudo_images_input \
    --pseudo_excelfile '/data/diag/rubenvdw/Info_files_Dataset_split/Pseudo_frames_excludeguidingTrue_excludeartefactFalse.xlsx'