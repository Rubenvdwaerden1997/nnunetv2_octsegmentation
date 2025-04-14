#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag\ \
#SBATCH --container-image="dodrio1.umcn.nl#rubenvdw/nnunetv2:1.3"
#SBATCH --qos=high
#SBATCH -o slurm_output/_slurm_output_predict_%j.txt
#SBATCH -e slurm_output/_slurm_error_predict_%j.txt
#SBATCH --exclude=dlc-nidoking,dlc-mewtwo,dlc-electabuzz,dlc-articuno,dlc-moltres,dlc-lugia

cd /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Pipeline/Faster_pipeline/Pipeline

# Print the hostname (GPU node) the job is running on
echo "Running on GPU node: $(hostname)"

python3 -u Pipeline_withoutinternalsaving.py \
    --data_input /data/diag/rubenvdw/Dataset/OCT_Case_Joske/DICOMS/ \
    --data_output /data/diag/rubenvdw/Dataset/OCT_Case_Joske/SEGMENTATIONS3/ \
    --k 3 \
    --radius 352 \
    --preprocessing  \
    --predict \
    --postprocessing_connected_component_dict /data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info/Pixels_postprocessing.txt \