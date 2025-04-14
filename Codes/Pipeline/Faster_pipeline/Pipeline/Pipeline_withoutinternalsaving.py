import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import time
import gc
import pydicom
import ast
#sys.path.insert(1, r'A:\6. Promovendi- onderzoekers\Volleberg Rick\Codes\nnunetv2\utils')
sys.path.insert(1, r'/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/Pipeline/Faster_pipeline/utils')
from conversion_utils import sample_around, create_circular_mask, rgb_to_grayscale_with_mapping
from predict import predict_from_files
from postprocessing import pixel_post_one


default_num_processes = 8 

def apply_connected_component_analysis(dict_loc):
    with open(dict_loc) as f:
        myData = f.read()
    classesPixels = ast.literal_eval(myData)

    return classesPixels

def preprocess_data(args, full_input_filename):
    id = 0
    processed_files = set()  # To track files that are already processed
    #Frames we want to sample around annotation 
    print(f'We are sampling {args.k} frames before and after each annotation')
    n_pullback = 0
    file_name = full_input_filename.split('\\')[-1]
    # Remove the .dcm extension
    pullback_name = file_name.replace('.dcm', '')
    # Find all related files

    print('Reading pullback: ', pullback_name)
    #Load the input files to create a list of slices
    print('Loading DICOM...')
    time_read_dicom = time.time()
    # series_inputdata = sitk.ReadImage(full_input_filename)
    # series_array_inputdata = sitk.GetArrayFromImage(series_inputdata)
    series_inputdata = pydicom.dcmread(full_input_filename)
    series_array_inputdata = series_inputdata.pixel_array
    time_end_read_dicom = time.time()
    print('Time elapsed reading DICOM: ', time_end_read_dicom - time_read_dicom)
    circular_mask_dcm = create_circular_mask(series_array_inputdata.shape[1], series_array_inputdata.shape[2], channels=series_array_inputdata.shape[3])
    masked_series_array_inputdata = np.invert(circular_mask_dcm) * series_array_inputdata
    masked_series_array_inputdata_gray = rgb_to_grayscale_with_mapping(masked_series_array_inputdata)

    frames_list = range(len(masked_series_array_inputdata_gray))
    frames_array_inputimage = np.zeros((len(masked_series_array_inputdata_gray), 704, 704,  (2*args.k+1)), dtype=np.float32)
    
    time_start = time.time()
    for frame in range(len(masked_series_array_inputdata_gray)):
        count = 0
        frames_around = sample_around(masked_series_array_inputdata_gray[:,:,:,0], frame, args.k)
        # I think this for loop is doing nothing now, so could be deleted! (?)
        for new_frame in range(frames_around.shape[2]):           
            if np.isnan(frames_around[:,:,new_frame]).any():
                raise ValueError('NaN detected')
            frames_array_inputimage[frame, :, :, :] = frames_around[:, :, :].astype(np.float32)        
            count += 1

    time_end = time.time()

    print(f'Done, time elapsed: {(time_end - time_start)}. Saved {len(frames_list)} frames from pullback {pullback_name} \n')
    print('###########################################\n')
    gc.collect()

    return frames_array_inputimage

def predict(frames_array_input_image, output_folder_file, k, predictor_reinitiate, postprocessing_connected_component_dict=None, predictor_used=None, num_processes=default_num_processes, batch_size=1, save_intermediate=False, save_intermediate_path=None, save_intermediate_name=None, save_intermediate_suffix=None, save_intermediate_compression=False, save_intermediate_compression_level=0, save_intermediate_compression_type='gzip', save_intermediate_compression_options=None, save_intermediate_compression_options_dict=None, save_intermediate_compression_options_dict_key=None, save_intermediate_compression_options_dict_value=None):
    # Placeholder for the prediction function
    print('Running predictions...')
    #Make sure that the input image is in the right shape (No. of frames, channels, height, width)
    if frames_array_input_image.shape[1] == 704 and frames_array_input_image.shape[2] == 704 and frames_array_input_image.shape[3] == (k*2+1):
        # Reshape to (375, 7, 704, 704) keeping the last two dimensions as 704 and 704
        reshaped_array = frames_array_input_image.transpose(0, 3, 1, 2)
    elif frames_array_input_image.shape[1] == 704 and frames_array_input_image.shape[2] == 704 and frames_array_input_image.shape[0] == (k*2+1):
        # Reshape to (375, 7, 704, 704) keeping the last two dimensions as 704 and 704
        reshaped_array = frames_array_input_image.transpose(3, 0, 1, 2)        
    else: 
        print(f'Input image might not be in correct shape. Should be e.g. (375, 7, 704, 704) but is in shape: {frames_array_input_image.shape}')
        print(f'File is not predicted, please check the input shape! Would have been saved as {output_folder_file} \n')
        return
    
    time_start = time.time()
    prediction_results, predictor_used = predict_from_files(reshaped_array, predictor_reinitiate)
    time_end = time.time()
    print(f'Prediction time: {time_end - time_start} seconds')

    if postprocessing_connected_component_dict is not None:
        classesPixels = apply_connected_component_analysis(postprocessing_connected_component_dict)
        time_start = time.time()
        # Postprocess the prediction results
        prediction_results_postprocessed = pixel_post_one(prediction_results, classesPixels, comb_thr=0, numClasses=14)
        time_end = time.time()
        print(f'Postprocessing time: {time_end - time_start} seconds')
        prediction_image = sitk.GetImageFromArray(prediction_results_postprocessed)
    else:
        prediction_image = sitk.GetImageFromArray(prediction_results)

    # Save the image as a .nii.gz file
    sitk.WriteImage(prediction_image, output_folder_file)

    print(f"Prediction results saved to {output_folder_file}!")

    return predictor_used


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Script for preprocessing and prediction")
        parser.add_argument('--data_input', type=str, default=r'A:\6. Promovendi- onderzoekers\Volleberg Rick\Codes\Test_image', help="Path to input folder.")
        parser.add_argument('--data_output', type=str, default=r'A:\6. Promovendi- onderzoekers\Volleberg Rick\Codes\Test_image_output', help="Path to output folder.")
        parser.add_argument('--k', type=int, default=3, help="Number of frames to sample around.")
        parser.add_argument('--radius', type=int, default=352, help="Radius for processing.")
        parser.add_argument('--preprocessing', action='store_true', help="Enable preprocessing.")
        parser.add_argument('--predict', action='store_true', help="Enable prediction.")
        parser.add_argument("--postprocessing_connected_component_dict", type=str, default="W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info/Pixels_postprocessing.txt")
    
        # Parse arguments
        args = parser.parse_args() #argv should be inside this and delete following lines

        input_folder = args.data_input
        output_folder = args.data_output
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Output folder '{output_folder}' created.")
        dcm_files = [f for f in os.listdir(input_folder) if f.endswith('.dcm')]

        if not dcm_files:
            print("No .dcm files found in the specified folder.")

        print(f"Found {len(dcm_files)} .dcm files. Starting processing...")

        # Set predictor to None, it will be initialized in the first iteration
        predictor_reinitiate = None
        # Run preprocessing if flag is set
        for dcm_file in dcm_files:
            time_start_loop = time.time()
            full_file_path = os.path.join(input_folder, dcm_file)
            output_file_path = os.path.join(output_folder, dcm_file.replace('.dcm', '_predictions.nii.gz'))
            print(f"Processing: {full_file_path}")
            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists, skipping processing.")
                continue  # Skip processing if the file already exists
            # Run preprocessing if flag is set
            if args.preprocessing:
                time_start = time.time()
                frames_array = preprocess_data(args, full_file_path)
                time_end = time.time()
                print(f"Preprocessing time: {time_end - time_start} seconds")
                print("Preprocessing complete, frames processed.")

            # Run prediction if flag is set
            if args.predict:
                time_start = time.time()
                predictor_reinitiate = predict(frames_array, output_file_path, args.k, predictor_reinitiate, args.postprocessing_connected_component_dict)
                time_end = time.time()
                print(f"Prediction time: {time_end - time_start} seconds")
                print("Prediction complete, results saved.")
            time_end_loop = time.time()
            print(f"Total processing time for {dcm_file}: {time_end_loop - time_start_loop} seconds")
            print("###########################################\n")
            
    except SystemExit as e:
        print(f"Error: {e}")
        sys.exit(2)  # Exit with error code 2