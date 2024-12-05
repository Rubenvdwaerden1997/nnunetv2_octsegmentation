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
    parser.add_argument('--outfolder', type=str, default='Z:/rubenvdw/nnunetv2/nnUnet/nnunetv2/Data/nnUNet_raw')
    parser.add_argument('--task', type=str, default='Pseudo_labels_complete_trainingset')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--radius', type=int, default=352)
    parser.add_argument('--splitfile', type=str, default='Z:/rubenvdw/Info_files_Dataset_split/15_classes_dataset_newsplit_29102024.xlsx')
    parser.add_argument('--allpullbacks_pectus', action='store_true', help="Use all pullbacks from the pectus dataset.")
    args, _ = parser.parse_known_args(argv)

    input_data_folders = args.data_input
    
    annots = pd.read_excel(args.splitfile)
    
    #Frames we want to sample around annotation 
    print('We are sampling {} frames before and after each annotation'.format(args.k))
    for input_data_folder in input_data_folders:
        input_data_folder_files = os.listdir(input_data_folder)
        for file_sep in input_data_folder_files:
            full_filename_inputdata=os.path.join(input_data_folder,file_sep)
            #Get file data from the metadata Excel file
            patient_name = "-".join(file_sep.split('.')[0].split('-')[:3])
            pullback_name = file_sep.split('.')[0]
            if not args.allpullbacks_pectus:
                if pullback_name not in annots['Pullback'].values:
                    print(f'Pullback: {pullback_name}, not in current training set')
                    continue
            
            # If patient name is not in annots['Patient'], it means that the patient is not in the training set
            belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]

            if belonging_set == 'Testing':
                print(f'Patient: {patient_name}, not in current training set but is present in {belonging_set} set, so not included as pseudo-labels')
                continue

            output_file_path_input = f'{args.outfolder}/{args.task}'
            os.makedirs(output_file_path_input, exist_ok=True)

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
            #Get frames with annotations in the pullback
            frames_with_annot = annots.loc[annots['Pullback'] == pullback_name]['Frames']
            frames_list = [int(i)-1 for i in frames_with_annot.values[0].split(',')]

            #RGB case
            time_start = time.time()
            for frame in range(len(masked_series_array_inputdata_gray)):
                if frame not in frames_list:
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

            time_end = time.time()

            print(f'Done, time elapsed: {(time_end - time_start)}. Saved {len(masked_series_array_inputdata_gray)-len(frames_list)} frames from pullback {pullback_name} \n')
            print('###########################################\n')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)