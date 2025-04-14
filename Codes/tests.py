import os
import glob
import SimpleITK as sitk
import sys
sys.path.insert(1, 'W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils')
from postprocessing import create_annotations_lipid, create_annotations_calcium, intima_cap_area_v2

# Define the directory
folder_path = r'W:\rubenvdw\Dataset\Pseudo_labels_output'

# Get all files matching the pattern
folder_path = r'W:\rubenvdw\Dataset\Pseudo_labels_output'

# Get all files in the folder
all_files = os.listdir(folder_path)

# Filter files that start with "NLDISALA0065_2_frame" and end with ".nii" or ".nii.gz"
nii_files = [
    os.path.join(folder_path, f)
    for f in all_files
    if f.startswith("NLDISALA0065_2_frame") and (f.endswith(".nii") or f.endswith(".nii.gz"))
]

# Function for analysis
def process_image(file_path):
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)[0]

    _, _, cap_thickness, lipid_arc, _ = create_annotations_lipid(image_array, font='mine')
    #print(f"Processed {file_path} | Cap Thickness: {cap_thickness} | Lipid Arc: {lipid_arc}")
    return file_path, cap_thickness, lipid_arc

# Process all matching files
results = []
for file in nii_files:
    try:
        result = process_image(file)
        results.append(result)
        #print(f"Processed: {file} | Cap Thickness: {result[1]} | Lipid Arc: {result[2]}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Optionally save results to a CSV
import pandas as pd
df = pd.DataFrame(results, columns=['Filename', 'Cap Thickness', 'Lipid Arc'])
df.to_csv(os.path.join(folder_path, 'lipid_analysis_results.csv'), index=False)
