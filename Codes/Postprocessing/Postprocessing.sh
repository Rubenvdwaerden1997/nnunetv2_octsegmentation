#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="doduo2.umcn.nl#rubenvdw/train_monai:v1.3"
#SBATCH -o _slurm_output_postprocessing_%j.txt
#SBATCH -e _slurm_error_postprocessing_%j.txt
#SBATCH --qos=high

python3 /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Postprocessing/Postprocessing.py processing \
    --input_folder /data/diag/rubenvdw/Dataset/Output_DICOMS_15classes \
    --output_folder /data/diag/rubenvdw/Dataset/Output_DICOMS_15classes_postprocessed_09122024 \
    --dict_loc /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info/Pixels_postprocessing.txt \
    --comb_thr 0 \