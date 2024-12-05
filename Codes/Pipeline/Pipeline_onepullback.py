import subprocess
import os

# Base directory for environment variables
BASE_DIR = r"/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data"

# Set environment variables
def setup_environment():
    print("\nSetting up environment variables...")
    os.environ["nnUNet_raw"] = f"{BASE_DIR}/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = f"{BASE_DIR}/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = f"{BASE_DIR}/nnUNet_results"
    
    # Optional: Verify the variables are set correctly
    print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# List of scripts with their respective arguments
scripts = [
    {
        "script": r"/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Dataset_conversion/Dataset_conversion_imgs_segs.py",
        "args": [
            "--data_input", r"/data/diag/rubenvdw/Dataset/Test_pullback predictions/Input",
            "--k", "3",
            "--radius", "352",
            "--preprocessing"
        ]
    },
    {
        "script": r"/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/nnunetv2/inference/predict_from_raw_data.py",
        "args": [
            "-i", r"/data/diag/rubenvdw/Dataset/Test_pullback predictions/Input/images_temp",
            "-o", r"/data/diag/rubenvdw/Dataset/Test_pullback predictions/Output",
            "-d", "601",
            "-c", "2d",
            "--save_probabilities",
        ]
    }
]

setup_environment()

# Sequentially execute each script with its arguments
for script_config in scripts:
    script_config=scripts[1]
    script = script_config["script"]
    args = script_config["args"]
    
    # Construct the command
    command = ["python", script] + args
    
    print(f"Running: {' '.join(command)}")
    
    # Execute the script
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script}: {e}")
        break  # Stop the sequence if an error occurs