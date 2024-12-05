import SimpleITK as sitk
import os
from natsort import natsorted
import numpy as np
import argparse

def concatenate_nii_files(folder_path, output_path, file_sep,file_extension=".nii", concatenate_axis=0):
    # Sort and read all files with the specified extension in the folder
    nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_extension) or f.endswith(file_extension + '.gz')]
    nii_files = natsorted(nii_files)  # Natural sorting of filenames
    
    # Generate a custom file name based on the first file in the list
    file_format = f'{file_sep.split(".")[0]}_HPNV.nii.gz'

    # Read and concatenate images using SimpleITK
    images = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in nii_files]
    concatenated_array = np.concatenate(images, axis=concatenate_axis)  # Concatenate along the specified dimension

    # Convert the concatenated array back to a SimpleITK image
    concatenated_image = sitk.GetImageFromArray(concatenated_array)

    # Set the origin, spacing, and direction from the first image (metadata)
    first_image = sitk.ReadImage(nii_files[0])
    concatenated_image.SetOrigin(first_image.GetOrigin())
    concatenated_image.SetSpacing(first_image.GetSpacing())
    concatenated_image.SetDirection(first_image.GetDirection())

    # Save the concatenated image with the custom file name
    sitk.WriteImage(concatenated_image, os.path.join(output_path, file_format))
    print(f"Concatenated image saved to: {os.path.join(output_path, file_format)}")

    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Concatenate .nii files along the specified dimension")
    
    # Use optional arguments (with '--')
    parser.add_argument('--folder_path', type=str, default=r'Z:\rubenvdw\Dataset\Test_pullback_predictions\tempfolder1', help="Path to the folder containing the .nii files")
    parser.add_argument('--output_path', type=str, default=r'Z:\rubenvdw\Dataset\Test_pullback_predictions\Output', help="Path to the output folder where the concatenated file will be saved")
    parser.add_argument('--file_sep', type=str, default='EST-NEMC-0005-LAD.dcm', help="Seperate file name, used to save correct filename")
    parser.add_argument('--file_extension', type=str, default=".nii", help="File extension for the .nii files (default: .nii)")
    parser.add_argument('--concatenate_axis', type=int, default=0, choices=[0, 1, 2], help="Axis to concatenate along (default: 0)")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    concatenate_nii_files(args.folder_path, args.output_path, args.file_sep,args.file_extension, args.concatenate_axis)

if __name__ == "__main__":
    main()