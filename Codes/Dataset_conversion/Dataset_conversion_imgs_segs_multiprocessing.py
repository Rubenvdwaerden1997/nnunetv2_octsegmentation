import sys
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse
import time
from multiprocessing import Pool
sys.path.insert(1, "/data/diag/rubenvdw/nnU-net/utils")
from conversion_utils import sample_around, resize_image
sys.path.insert(1, "/data/diag/rubenvdw/MONAI/utils")
from utils_conversion import create_circular_mask
from rgb_to_gray_mapping import rgb_to_grayscale_with_mapping

def process_pullback(params):
    input_data_folder, label_data, pullback_name, args, annots, frames_list = params

    full_filename_inputdata = os.path.join(input_data_folder, pullback_name)
    full_filename_labeldata = os.path.join(label_data, pullback_name.replace('.dcm', '_HPNV.nii.gz'))

    # Metadata extraction
    patient_name = "-".join(pullback_name.split('.')[0].split('-')[:3])
    belonging_set = annots.loc[annots['Patient'] == patient_name]['Set'].values[0]
    output_file_path_input = f"{args.outfolder}/{args.task}/images{'Ts' if belonging_set == 'Testing' else 'Tr'}"
    output_file_path_label = f"{args.outfolder}/{args.task}/labels{'Ts' if belonging_set == 'Testing' else 'Tr'}"
    os.makedirs(output_file_path_input, exist_ok=True)
    os.makedirs(output_file_path_label, exist_ok=True)

    # Load input DICOM
    series_inputdata = sitk.ReadImage(full_filename_inputdata)
    series_array_inputdata = sitk.GetArrayFromImage(series_inputdata)
    circular_mask_dcm = create_circular_mask(series_array_inputdata.shape[1], series_array_inputdata.shape[2], channels=series_array_inputdata.shape[3])
    masked_series_array_inputdata = np.invert(circular_mask_dcm) * series_array_inputdata
    masked_series_array_inputdata_gray = rgb_to_grayscale_with_mapping(masked_series_array_inputdata)

    # Load label
    series_labeldata = sitk.ReadImage(full_filename_labeldata)
    series_array_labeldata = sitk.GetArrayFromImage(series_labeldata)

    if args.preprocessing:
        # Label preprocessing
        series_array_labeldata[series_array_labeldata == 11] = 3
        series_array_labeldata[series_array_labeldata == 12] = 11
        series_array_labeldata[series_array_labeldata == 13] = 12
        series_array_labeldata[series_array_labeldata == 14] = 13

    # Process frames
    for frame in range(len(masked_series_array_inputdata_gray)):
        if frame in frames_list:
            frames_around = sample_around(masked_series_array_inputdata_gray[:, :, :, 0], frame, args.k)
            for new_frame in range(frames_around.shape[2]):
                final_path_input = f"{output_file_path_input}/{patient_name.replace('-', '')}_{frame}_{new_frame}.nii"
                if not os.path.exists(final_path_input):
                    frame_to_save = np.zeros((1, 704, 704), dtype=np.float32)
                    frame_to_save[0, :, :] = frames_around[:, :, new_frame]
                    final_image_after = sitk.GetImageFromArray(frame_to_save)
                    final_image_after.SetSpacing((1.0, 1.0, 999.0))
                    sitk.WriteImage(final_image_after, final_path_input)

            # Save segmentation
            final_path_label = f"{output_file_path_label}/{patient_name.replace('-', '')}_{frame}.nii"
            if not os.path.exists(final_path_label):
                resampled_seg_frame = resize_image(series_array_labeldata[frame, :, :])
                circular_mask = create_circular_mask(resampled_seg_frame.shape[0], resampled_seg_frame.shape[1], radius=args.radius, channels=0)
                masked_resampled_frame = np.invert(circular_mask) * resampled_seg_frame
                final_array = np.zeros((1, 704, 704), dtype=np.uint8)
                final_array[0, :, :] = masked_resampled_frame
                final_frame = sitk.GetImageFromArray(final_array)
                final_frame.SetSpacing((1.0, 1.0, 999.0))
                sitk.WriteImage(final_frame, final_path_label)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input', type=str, nargs='+', default=['Z:/rubenvdw/Dataset/DICOMS_15classes', 'Z:/rubenvdw/Dataset/DICOMS_Orange'])
    parser.add_argument('--data_label', type=str, default='Z:/rubenvdw/Dataset/SEGMENTATIONS_15classes_25102024')
    parser.add_argument('--outfolder', type=str, default='Z:/rubenvdw/nnunetv2/nnUnet/nnunetv2/Data/nnUNet_raw')
    parser.add_argument('--task', type=str, default='Dataset601')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--radius', type=int, default=352)
    parser.add_argument('--splitfile', type=str, default='Z:/rubenvdw/Info_files_Dataset_split/15_classes_dataset_split_extraframes_25102024.xlsx')
    parser.add_argument('--preprocessing', action='store_true')
    args, _ = parser.parse_known_args(argv)

    annots = pd.read_excel(args.splitfile)
    input_data_folders = args.data_input
    label_data = args.data_label

    params = []
    for input_data_folder in input_data_folders:
        input_data_folder_files = os.listdir(input_data_folder)
        for pullback_name in input_data_folder_files:
            if pullback_name.split('.')[0] not in annots['Pullback'].values:
                continue
            frames_with_annot = annots.loc[annots['Pullback'] == pullback_name.split('.')[0]]['Frames']
            frames_list = [int(i) - 1 for i in frames_with_annot.values[0].split(',')]
            params.append((input_data_folder, label_data, pullback_name, args, annots, frames_list))

    # Multiprocessing Pool
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_pullback, params)

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)