import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import argparse
import sys
import ast
from pathlib import Path
import torch

import skimage as ski

##########################
# From here are the functions for postprocessing
##########################
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


def pixel_post_one(filename, classesPixels, comb_thr, numClasses):
    # Check if filename is already an array
    if isinstance(filename, torch.Tensor):
        orig_seg_pixel_array = filename
        if orig_seg_pixel_array.dim() == 2:  # If it has shape (H, W), add a new first dimension
            orig_seg_pixel_array = orig_seg_pixel_array.unsqueeze(0)
    elif isinstance(filename, np.ndarray):
        orig_seg_pixel_array = filename
        if orig_seg_pixel_array.ndim == 2:  # If it has shape (H, W), add a new first dimension
            orig_seg_pixel_array = np.expand_dims(orig_seg_pixel_array, axis=0)
    else:
        # Load the image if filename is not an array
        orig_seg = sitk.ReadImage(filename)
        orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)

    if comb_thr == 1:
        orig_seg_pixel_array[orig_seg_pixel_array==9] = 10

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

def pixel_post_many(annotations_folder_path, output_folder_path, classesPixels, comb_thr, onlyN2, numClasses): #onlyN2 needed for function signature
    # Create output folder if it does not exist
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    filenames = os.listdir(annotations_folder_path)
    for filename in filenames:
        if filename.endswith('.nii.gz') or filename.endswith('.nii'):
    
            #Need to change later. Need to get all filenames one by one
            file_path = f'{annotations_folder_path}/{filename}'
    
            final_array = pixel_post_one(file_path, classesPixels, comb_thr, numClasses)
    
            #Need to change later
            final_path = f'{output_folder_path}/{filename}'
    
            #Correct spacing and direction and save as nifti
            final_frame = sitk.GetImageFromArray(final_array.astype(np.uint32))
            final_frame.SetSpacing((1.0, 1.0, 999.0))
            final_frame.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            sitk.WriteImage(final_frame, final_path)

##########################
# From here are the postprocessing functions for doing all at once for 13-class article
##########################

def pixel_post_all(annotations_folder_path, output_folder_path, classesPixels, comb_thr, onlyN2, numClasses):
    #Find all subfolders in annotations folder path, make those folders in output folder path. Run postprocessing on all combis
    names1 = ['fold_0','fold_1','fold_2','fold_3','fold_4',]
    names1_2 = ['model_199','model_399','model_599','model_799','model_best','model_final_checkpoint']
    names2 = ['testSetBest','testSetLast']
    if onlyN2 == 0:
        for name1 in names1:
            for name1_2 in names1_2:
                newPath_in = f'{annotations_folder_path}/{name1}/predictions/{name1_2}'
                newPath_out = f'{output_folder_path}/{name1}/predictions/{name1_2}'
                Path(newPath_out).mkdir(parents=True, exist_ok=True)
                pixel_post_many(newPath_in, newPath_out, classesPixels, comb_thr, onlyN2, numClasses)
    for name2 in names2:
        newPath_in = f'{annotations_folder_path}/{name2}'
        newPath_out = f'{output_folder_path}/{name2}'
        Path(newPath_out).mkdir(parents=True, exist_ok=True)
        pixel_post_many(newPath_in, newPath_out, classesPixels, comb_thr, onlyN2, numClasses)

##########################
# From here are the functions for counting
##########################
def countClass(myArray, class_ind, num_jumps):
    if class_ind not in myArray[0,:,:]:
        return np.nan, np.nan
    
    temp_array = np.copy(myArray)[0]
    
    if class_ind == 0:
        temp_array[temp_array==0] = 20
        temp_array[temp_array<20] = 0
        temp_array[temp_array==20] = 1
    else:
        temp_array[temp_array!=class_ind] = 0
        temp_array[temp_array==class_ind] = 1
        
    classTotal = np.sum(temp_array)
    labeled_image, count = ski.measure.label(temp_array, connectivity=num_jumps, return_num=True)
    unique, counts = np.unique(labeled_image, return_counts=True)
    return np.min(counts[1:]), classTotal
    
    
    
def countOne(myArray, myDict):
    numClasses = len(myDict.items())
    counts = np.ones(numClasses*2)
    for item in myDict.items():
        classMin, classTotal = countClass(myArray, item[0], item[1][1])
        counts[item[0]] = classMin
        counts[item[0]+numClasses]  = classTotal
    return counts
    

def countAll(ann_path, out_path, myDict, comb_thr, onlyN2, dummy): #onlyN2 needed for signature of function
    filenames = os.listdir(ann_path)
    numClasses = len(myDict.items())
    allValues = []
    
    classes = ['background', 'lumen', 'guidewire','intima','lipid','calcium','media','catheter','sidebranch',
              'red thrombus', 'white thrombus', 'dissection', 'plaque rupture', 'healed plaque', 'neovascularization']
    
    classes = classes[:numClasses]
    
    totals = ['background_t', 'lumen_t', 'guidewire_t','intima_t','lipid_t','calcium_t','media_t','catheter_t','sidebranch_t',
              'red thrombus_t', 'white thrombus_t', 'dissection_t', 'plaque rupture_t', 'healed plaque_t', 'neovascularization_t']
    
    minima = [np.nan for i in range(numClasses)]
    
    allRows = []
    
    for filename in filenames:
        if filename.endswith('.nii.gz'):         
            orig_seg = sitk.ReadImage(f'{ann_path}/{filename}')
            orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)
            amounts = countOne(orig_seg_pixel_array, myDict)
            allRows.append(filename)
            allValues.append(amounts)
            
            for i in range(numClasses):
                if np.isnan(amounts[i]):
                    continue
                elif np.isnan(minima[i]):
                    minima[i] = amounts[i]
                elif minima[i]>amounts[i]:
                    minima[i] = amounts[i]
                    
    
    myDF_specifics = pd.DataFrame([minima], columns=classes)

    classes.extend(totals[:numClasses])

    myDF_all = pd.DataFrame(allValues, columns=classes)
    
    myDF_all.index = allRows

    out_filePath1 = f'{out_path}/allMinimaPerFrame.xlsx'
    out_filePath2 = f'{out_path}/total_metrics.xlsx'
    
    myDF_all.to_excel(out_filePath1)
    myDF_specifics.to_excel(out_filePath2)

##########################
# From here is the main activation
##########################

def init(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files')
    parser.add_argument('--output_folder', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files_postprocessing')
    parser.add_argument('--dict_loc', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info/Pixels_postprocessing.txt')
    parser.add_argument('--comb_thr', type=int, default=0)
    parser.add_argument('--only_n2', type=int, default=0)
    parser.add_argument('--numClasses', type=int, default=15)
    args, _ = parser.parse_known_args(argv)

    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output folder ready: {args.output_folder}")
    
    return args


if __name__ == '__main__':
    # Very first argument determines action
    actions = {
        'counting': countAll,
        'processing': pixel_post_many,
        'processall': pixel_post_all
    }

    try:
        action = actions[sys.argv[1]]
        action = actions[sys.argv[1]]
        args = init(sys.argv[2:])
    except (IndexError, KeyError):
        print('Usage: nnunet ' + '/'.join(actions.keys()) + ' ...')
    else:
        with open(args.dict_loc) as f:
            myData = f.read()

        classesPixels = ast.literal_eval(myData)

        action(args.input_folder,args.output_folder, classesPixels, args.comb_thr, args.only_n2, args.numClasses)
