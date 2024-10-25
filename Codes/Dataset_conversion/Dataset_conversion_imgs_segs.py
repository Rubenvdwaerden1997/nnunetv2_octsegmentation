import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import time
sys.path.insert(1, "/data/diag/rubenvdw/nnU-net/utils")
from conversion_utils import sample_around, resize_image
sys.path.insert(1, "/data/diag/rubenvdw/MONAI/utils")
from utils_conversion import create_circular_mask
from rgb_to_gray_mapping import rgb_to_grayscale_with_mapping

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input', type=str, nargs='+', default=['Z:/rubenvdw/Dataset/DICOMS_15classes', 'Z:/rubenvdw/Dataset/DICOMS_Orange'], help="Paths to one or more data directories.")
    parser.add_argument('--data_label', type=str, default='Z:/rubenvdw/Dataset/SEGMENTATIONS_15_classes')
    parser.add_argument('--outfolder', type=str, default='Z:/rubenvdw/nnunetv2/nnUnet/nnunetv2/Data/nnUNet_raw')
    parser.add_argument('--task', type=str, default='Dataset601')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--radius', type=int, default=352)
    parser.add_argument('--splitfile', type=str, default='Z:/rubenvdw/Info_files_Dataset_split/15_classes_dataset_split_extraframes_25102024.xlsx')
    args, _ = parser.parse_known_args(argv)

    input_data_folders = args.data_input
    label_data = args.data_label
    
    annots = pd.read_excel(args.splitfile)
    
    #Frames we want to sample around annotation 
    print('We are sampling {} frames before and after each annotation'.format(args.k))
    for input_data_folder in input_data_folders:
        input_data_folder_files = os.listdir(input_data_folder)
        for file_sep in input_data_folder_files:
            full_filename_inputdata=os.path.join(input_data_folder,file_sep)
            full_filename_labeldata=os.path.join(label_data,file_sep.replace('.dcm','_HPNV.nii.gz'))
            #Get file data from the metadata Excel file
            patient_name = "-".join(file_sep.split('.')[0].split('-')[:3])
            pullback_name = file_sep.split('.')[0]
            if pullback_name not in annots['Pullback'].values:
                print(f'Pullback: {pullback_name}, not in current training set')
                continue
            
            # If patient name is not in annots['Patient'], it means that the patient is not in the training set
            belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

            #Output folder and create folder if needed
            if belonging_set == 'Testing':
                output_file_path_input = f'{args.outfolder}/{args.task}/imagesTs'
                output_file_path_label = f'{args.outfolder}/{args.task}/labelsTs'
            else:
                output_file_path_input = f'{args.outfolder}/{args.task}/imagesTr'
                output_file_path_label = f'{args.outfolder}/{args.task}/labelsTr'
            os.makedirs(output_file_path_input, exist_ok=True)
            os.makedirs(output_file_path_label, exist_ok=True)

            #More metadata
            id = int(annots.loc[annots['Patient'] == patient_name]['ID'].values[0])

            n_pullback = int(annots.loc[annots['Pullback'] == pullback_name]['Nº pullback'].values[0])
            print('Reading pullback: ', pullback_name)
            #Load the input files to create a list of slices
            print('Loading DICOM...')
            time_read_dicom = time.time()
            series_inputdata = sitk.ReadImage(full_filename_inputdata)
            series_array_inputdata = sitk.GetArrayFromImage(series_inputdata)
            time_end_read_dicom = time.time()
            print('Time elapsed reading DICOM: ', time_end_read_dicom - time_read_dicom)
            circular_mask_dcm = create_circular_mask(series_array_inputdata.shape[1], series_array_inputdata.shape[2], channels=series_array_inputdata.shape[3])
            masked_series_array_inputdata = np.invert(circular_mask_dcm) * series_array_inputdata
            masked_series_array_inputdata_gray = rgb_to_grayscale_with_mapping(masked_series_array_inputdata)
            #Load the label files to create a list of slices
            print('Loading Segmentation...')
            time_read_seg = time.time()
            series_labeldata = sitk.ReadImage(full_filename_labeldata)
            series_array_labeldata = sitk.GetArrayFromImage(series_labeldata)
            time_end_read_seg = time.time()
            print('Time elapsed reading Segmentation: ', time_end_read_seg - time_read_seg)

            #Get frames with annotations in the pullback
            frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
            frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

            #RGB case
            time_start = time.time()
            for frame in range(len(masked_series_array_inputdata_gray)):
                if frame in frames_list:
                    count = 0
                    frames_around = sample_around(masked_series_array_inputdata_gray[:,:,:,0], frame, args.k)
                    for new_frame in range(frames_around.shape[2]):           
                        final_path_input = output_file_path_input + '/' + patient_name.replace("-", "") + '_{}_frame{}_{}_{}.nii'.format(n_pullback, frame, "%03d" % id,"%04d" % (count))
                        if os.path.exists(final_path_input):
                            count += 1
                            print('File already exists')
                            continue
                        else:
                            if np.isnan(frames_around[:,:,new_frame]).any():
                                raise ValueError('NaN detected')
                            frame_to_save= np.zeros((1, 704, 704))
                            frame_to_save[0,:,:]=frames_around[:,:,new_frame].astype(np.float32)
                            final_image_after = sitk.GetImageFromArray(frame_to_save)
                            final_image_after.SetSpacing((1.0, 1.0, 999.0))
                            final_image_after.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(final_image_after, final_path_input)
                            count += 1
                    # Save segmentation
                    final_path_label = output_file_path_label + '/' + patient_name.replace('-', '') + '_{}_frame{}_{}.nii'.format(n_pullback, frame, "%03d" % id)

                    #Check that a seg has already been generated
                    if os.path.exists(final_path_label):
                        print('File already exists. Skip')
                    else:
                        raw_frame = series_array_labeldata[frame,:,:]
                        spacing = annots.loc[annots['Pullback'] == pullback_name]['Spacing'].values[0]

                        #Check if resize is neeeded (shape should be (704, 704) and with the correct spacing)
                        if raw_frame.shape == (1024, 1024) and spacing == 0.006842619:
                            resampled_seg_frame = resize_image(raw_frame)
                        elif raw_frame.shape == (1024, 1024) and spacing == 0.009775171:
                            resampled_seg_frame = raw_frame[160:864, 160:864]
                        elif raw_frame.shape == (704, 704) and (spacing == 0.014224751 or spacing == 0.014935988):
                            resampled_seg_frame = resize_image(raw_frame,False)
                            resampled_seg_frame = resampled_seg_frame[160:864, 160:864]
                        else:
                            resampled_seg_frame = raw_frame

                        #Apply mask to seg
                        circular_mask = create_circular_mask(resampled_seg_frame.shape[0], resampled_seg_frame.shape[1], radius=args.radius, channels=0)
                        masked_resampled_frame = np.invert(circular_mask) * resampled_seg_frame

                        #Sanity checks
                        if np.isnan(masked_resampled_frame).any():
                            raise ValueError('NaN detected')
                        
                        #Need to add extra dimension
                        final_array = np.zeros((1, 704, 704))
                        final_array[0,:,:] = masked_resampled_frame

                        #Correct spacing and direction and save as nifti
                        final_frame = sitk.GetImageFromArray(final_array.astype(np.uint8))
                        final_frame.SetSpacing((1.0, 1.0, 999.0))
                        final_frame.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(final_frame, final_path_label)

            time_end = time.time()

            print(f'Done, time elapsed: {(time_end - time_start)}. Saved {len(frames_list)} frames from pullback {pullback_name} \n')
            print('###########################################\n')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)