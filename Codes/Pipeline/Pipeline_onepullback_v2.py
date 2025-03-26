import subprocess
import os
import argparse
import time
import shutil
import natsort

# Set environment variables
def setup_environment(base_dir):
    print("\nSetting up environment variables...")
    os.environ["nnUNet_raw"] = f"{base_dir}/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = f"{base_dir}/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = f"{base_dir}/nnUNet_results"
    
    # Optional: Verify the variables are set correctly
    print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# Run the provided script with its arguments
def run_scripts(scripts):
    for script_config in scripts:
        script = script_config["script"]
        args = script_config["args"]
        
        # Construct the command
        command = ["python3", script] + args
        
        print(f"Running: {' '.join(command)}")
        
        # Execute the script
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while running {script}: {e}")
            break  # Stop the sequence if an error occurs

def delete_temp_folders(parent_folder):
    """Delete all temp folders and wait until they are fully deleted."""
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path) and item.startswith("tempfolder"):
            # Delete the folder
            shutil.rmtree(item_path)
            print(f"Deleting folder: {item_path}")
            time_start_deletion = time.time()
            # Wait until the folder is fully deleted
            while os.path.exists(item_path):
                time.sleep(10)  # Check every 10s
                time_end_deletion = time.time()
                print(f'Time elapsed: {time_end_deletion - time_start_deletion} seconds.')

            print(f"Deleted folder: {item_path}")

def main():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts with environment variables.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for environment variables.")
    parser.add_argument("--scripts", type=str, nargs="+", required=True,
                        help="Scripts and arguments. Format: script_path,arg1,arg2,...")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    
    args = parser.parse_args()
    print('Start pipeline')
    parent_folder = os.path.dirname(args.input_folder)
    os.makedirs(args.output_folder, exist_ok=True)  # Ensure output folder exists

    print(f'Parent folder: {parent_folder}')
    print(f'Output folder: {args.output_folder}')
    
    output_temp_folder = os.path.join(parent_folder, "tempfolder1")
    
    # Parse scripts and arguments
    delete_temp_folders(parent_folder)
    for file_sep in os.listdir(args.input_folder):
        if not file_sep.endswith(".dcm"):
            continue
        base_filename = file_sep.replace(".dcm", "")  # Remove .dcm
        expected_output_file = f"{base_filename}_HPNV.nii.gz"
        output_file_path = os.path.join(args.output_folder, expected_output_file)
        if os.path.exists(output_file_path):
            print(f"Skipping {file_sep}, already segmented as {expected_output_file}.")
            continue  # Move to the next file

        os.makedirs(output_temp_folder, exist_ok=True)
        scripts = []
        time_start = time.time()
        for i, script_entry in enumerate(args.scripts):
            parts = script_entry.split(",")
            script_path = parts[0]
            script_args = parts[1:]

            # Add '--file_sep' argument only for the first script entry
            if i == 0 or i ==2:
                script_args.append(f"--file_sep={file_sep}")
            
            scripts.append({"script": script_path, "args": script_args})
        
        setup_environment(args.base_dir)
        run_scripts(scripts)
        time_end = time.time()

        print(f'Finished file {file_sep}, in {time_end - time_start:.2f} seconds.')
        delete_temp_folders(parent_folder)

    print('Finished pipeline')
if __name__ == "__main__":
    main()