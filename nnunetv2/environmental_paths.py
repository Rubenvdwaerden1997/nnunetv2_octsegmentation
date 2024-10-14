import os
from pathlib import Path

# Base directory
base_dir = Path(r"/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data")

# Set environment variables for nnUNet directories based on the base directory
os.environ['nnUNet_raw'] = str(base_dir / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(base_dir / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(base_dir / 'nnUNet_results')

# Optional: Print the values to verify they are set correctly
print("nnUNet_raw:", os.environ["nnUNet_raw"])
print("nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
print("nnUNet_results:", os.environ["nnUNet_results"])
