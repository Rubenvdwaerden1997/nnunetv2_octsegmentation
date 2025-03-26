import SimpleITK as sitk
import os
from natsort import natsorted
import numpy as np
import argparse
import torch
import skimage as ski

def get_surrounding(coord_array):
    coords = np.zeros((coord_array.shape[0]*8,2))
    for ind,coord in enumerate(coord_array):
        coords[ind*8:ind*8+8] = get_surrounding_for1(coord)
    uniques = np.unique(coords,axis=0).tolist()
    
    #Lastly remove any cases that are out of bounds
    final_uniques = np.array([i for i in uniques if -1<i[0]<704 and -1<i[1]<704 and i not in coord_array.tolist()])
    return final_uniques

def get_surrounding_for1(coord):
    x,y = coord
    surrounding = np.array([[x-1,y],[x-1,y+1],[x-1,y-1],[x+1,y],[x+1,y+1],[x+1,y-1],[x,y-1],[x,y+1]])
    return surrounding

def pixel_post_one(orig_seg_pixel_array, classesPixels, numClasses):
    orig_seg_copy = np.copy(orig_seg_pixel_array)
    
    for frame in range(orig_seg_pixel_array.shape[0]): #So it will work for either full pullback or single frame
        for item in classesPixels.items(): 
            #Need to adapt to work with background
            class_ind = item[0]
            pixel_minimum = item[1][0]
            temp_array = np.copy(orig_seg_pixel_array[frame])
            if class_ind == 0:
                temp_array[temp_array==0] = 20
                temp_array[temp_array<20] = 0
                temp_array[temp_array==20] = 1
            else:
                temp_array[temp_array!=class_ind] = 0
                temp_array[temp_array==class_ind] = 1
        
            labeled_image, count = ski.measure.label(temp_array, connectivity=item[1][1], return_num=True)
        
            unique, counts = np.unique(labeled_image, return_counts=True)
        
            for group_ind in range(1,count+1):#Loop over the groups
                if counts[group_ind]>pixel_minimum:
                    continue
                else:
                    main_indices = np.where(labeled_image == group_ind)
                    usable_main_indices = np.array(list(zip(main_indices[0],main_indices[1])))
                    neighbour_indices = get_surrounding(usable_main_indices)
                
                    neighbour_indices = neighbour_indices.astype(int)
                
                    current_counts = np.zeros(numClasses)
                    #Loop over the neighbour indices and get counts of classes
                    for neighbour in neighbour_indices: 
                        current_counts[orig_seg_copy[frame,neighbour[0],neighbour[1]]] += 1
                
                    if (current_counts[0]/np.sum(current_counts)) > 0.75:
                        new_class = 0
                    else:
                        new_class = np.argmax(current_counts[1:])+1
                
                    #Will fail if only 1 index is found (is that really the case?)
                    for main in usable_main_indices:
                        orig_seg_copy[frame,main[0],main[1]] = new_class
                        
    return orig_seg_copy

def concatenate_nii_files(folder_path, output_path, file_sep,file_extension=".nii", concatenate_axis=0, postprocessing_dict=None):
    # Sort and read all files with the specified extension in the folder
    nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_extension) or f.endswith(file_extension + '.gz')]
    nii_files = natsorted(nii_files)  # Natural sorting of filenames
    
    # Generate a custom file name based on the first file in the list
    file_format = f'{file_sep.split(".")[0]}_HPNV'

    if file_extension==".nii" or file_extension==".nii.gz":
        file_format = f'{file_format}.nii.gz'
        # Read and concatenate images using SimpleITK
        images = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in nii_files]
        concatenated_array = np.concatenate(images, axis=concatenate_axis)  # Concatenate along the specified dimension
        if postprocessing_dict is not None:
            print(f'Postprocessing the concatenated image with the following dictionary: {postprocessing_dict}')
            concatenated_array = pixel_post_one(concatenated_array, postprocessing_dict, numClasses=14)
            
        # Convert the concatenated array back to a SimpleITK image
        concatenated_image = sitk.GetImageFromArray(concatenated_array)

        # Set the origin, spacing, and direction from the first image (metadata)
        first_image = sitk.ReadImage(nii_files[0])
        concatenated_image.SetOrigin(first_image.GetOrigin())
        concatenated_image.SetSpacing(first_image.GetSpacing())
        concatenated_image.SetDirection(first_image.GetDirection())

        # Save the concatenated image with the custom file name
        sitk.WriteImage(concatenated_image, os.path.join(output_path, file_format))
    elif file_extension==".npz":
        images = [np.load(f)['probabilities'] for f in nii_files] # shape is channels, 1, 704,704
        images = [np.moveaxis(img, 0, 1) for img in images]  # shape conversion to 1, channels, 704,704
        concatenated_array = np.concatenate(images, axis=concatenate_axis)

        np.savez_compressed(os.path.join(output_path, file_format), probabilities=concatenated_array)

    print(f"Concatenated image saved to: {os.path.join(output_path, file_format)}")
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Concatenate .nii files along the specified dimension")
    
    # Use optional arguments (with '--')
    parser.add_argument('--folder_path', type=str, default=r'W:\rubenvdw\Dataset\blab\predictions', help="Path to the folder containing the .nii files")
    parser.add_argument('--output_path', type=str, default=r'W:\rubenvdw\Dataset\blab', help="Path to the output folder where the concatenated file will be saved")
    parser.add_argument('--file_sep', type=str, default='ABW-ARU-0001.dcm', help="Seperate file name, used to save correct filename")
    parser.add_argument('--file_extension', type=str, default=".nii", help="File extension for the .nii files (default: .nii)")
    parser.add_argument('--concatenate_axis', type=int, default=0, choices=[0, 1, 2], help="Axis to concatenate along (default: 0)")
    parser.add_argument('--postprocessing', type=bool, default=True, help="Whether to apply postprocessing to the concatenated image (default: True)")

    # Parse the arguments
    args = parser.parse_args()
    if args.postprocessing:
        postprocessing_dict = {0: [300,1],
            1: [300,1],
            2: [1000,1],
            3: [1000,1],
            4: [1000,1],
            5: [100,1],
            6: [100,1],
            8: [100,1]
            }
    else:
        postprocessing_dict = None
        
    # Call the function with the provided arguments
    concatenate_nii_files(args.folder_path, args.output_path, args.file_sep,args.file_extension, args.concatenate_axis, postprocessing_dict)

if __name__ == "__main__":
    main()