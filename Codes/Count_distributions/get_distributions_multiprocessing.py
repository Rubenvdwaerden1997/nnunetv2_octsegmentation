import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import SimpleITK as sitk
import os
import argparse
import time  # For timing
from multiprocessing import Pool, cpu_count
import sys

# Import custom modules
sys.path.insert(1, '/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils')
from postprocessing import create_annotations_lipid, create_annotations_calcium, intima_cap_area_v2

class Get_Distributions:

    def __init__(self, data_paths: str, output_filename: str, data_info: str, num_classes: int, lipid_cap_area_manual: int, task_name: str):
        self.data_paths = data_paths
        self.output_filename = output_filename
        self.data_info = pd.read_excel('{}.xlsx'.format(data_info))
        self.num_classes = num_classes
        self.lipid_cap_area_manual = lipid_cap_area_manual
        self.task_name = task_name

    def get_patient_data(self, file: str) -> str:
        # Function to extract patient data from the filename
        filename = file.split('_')[0]
        n_pullback = file.split('_')[1]
        # Logic to get the patient name and pullback name
        if filename.startswith('NLRUMC'):
            patient_name = '{}-{}-{}'.format(filename[:2], filename[2:-5], filename[-5:])
        else:
            patient_name = '{}-{}-{}'.format(filename[:3], filename[3:-4], filename[-4:])
        pullback_name = self.data_info[(self.data_info['Nº pullback'] == int(n_pullback)) & (self.data_info['Patient'] == patient_name)]['Pullback'].values[0]
        return pullback_name

    def process_file(self, file: str):
        if not (file.endswith('nii.gz') or file.endswith('nii')):
            return None
        filename = os.path.basename(file)
        pullback_name = self.get_patient_data(filename)
        start_time = time.time()  # Start timing for this file
        print(f"Processing pullback: {pullback_name}")

        n_frame = int(filename.split('_')[2][5:])

        seg_map = sitk.ReadImage(file)
        seg_map_data = sitk.GetArrayFromImage(seg_map)[0]

        # Get count of labels in each frame
        one_hot = np.zeros(self.num_classes)
        unique = np.unique(seg_map_data).astype(int)
        one_hot[unique] = 1

        # Post-processing results
        _, _, cap_thickness, lipid_arc, _ = create_annotations_lipid(seg_map_data, font='cluster')
        _, _, calcium_depth, calcium_arc, calcium_thickness, _ = create_annotations_calcium(seg_map_data, font='cluster')

        # Lipid Cap area calculations
        manual_lipid_cap_area = 0
        if self.lipid_cap_area_manual == 1:
            manual_map = sitk.ReadImage(os.path.join('/data/diag/rubenvdw/nnU-net/data-2d/nnUNet_raw_data', self.task_name, 'labelsTs', file))
            manual_map_data = sitk.GetArrayFromImage(manual_map)[0]
            _, manual_lipid_cap_area = intima_cap_area_v2(manual_map_data)
            
        _, ai_lipid_cap_area = intima_cap_area_v2(seg_map_data)

        end_time = time.time()  # End timing for this file
        print(f"Completed processing pullback: {pullback_name}, frame: {n_frame} in {end_time - start_time:.2f} seconds")
        
        # Compile data for each frame
        return [pullback_name, n_frame] + one_hot.tolist() + [lipid_arc, cap_thickness, calcium_depth, calcium_arc, calcium_thickness, manual_lipid_cap_area, ai_lipid_cap_area]

    def get_counts(self):
        columns = ['Pullback', 'Frame'] + ['AI_background', 'AI_lumen', 'AI_guidewire', 'AI_intima', 'AI_lipid', 'AI_calcium',
                    'AI_media', 'AI_catheter', 'AI_sidebranch', 'AI_rthrombus', 'AI_wthrombus','AI_plaque_rupture',
                     'AI_layered_plaque','AI_neovascularization', 
                     'AI_lipid_arc', 'AI_FCT', 'AI_calcium_depth', 'AI_calcium_arc', 'AI_calcium_thickness','Manual_FC_Area', 'AI_FC_Area']

        print("Starting processing of files for counts...")
        start_time = time.time()  # Start timing for the whole get_counts process
        all_files = []
        for data_path in self.data_paths:
            all_files.extend([os.path.join(data_path, f) for f in os.listdir(data_path)])

        
        with Pool(cpu_count()) as pool:
            results = pool.map(self.process_file, all_files)

        # Before creating the DataFrame
        print(f"Number of columns defined: {len(columns)}")
        print(f"Sample data lengths: {[len(result) for result in results if result is not None]}")

        # Then create the DataFrame if lengths match, or inspect further if they don’t.
        if all(len(result) == len(columns) for result in results if result is not None):
            counts_per_frame = pd.DataFrame([result for result in results if result is not None], columns=columns)
        else:
            print("Mismatch between data and columns length.")
        counts_per_frame.to_excel('{}.xlsx'.format(self.output_filename), index=False)
        
        end_time = time.time()  # End timing for the whole get_counts process
        print(f"All pullbacks processed in {end_time - start_time:.2f} seconds. Results saved to {self.output_filename}.xlsx")


# Main function remains the same
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str, nargs='+', default=[r'Z:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Data\nnUNet_raw\Dataset701_TS3D3\labelsTr',r'Z:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Data\nnUNet_raw\Dataset701_TS3D3\labelsTr'])
    parser.add_argument('--output_filename', type=str, default=r'Z:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Predictions\Dataset_dummy\Metrics\Dummy_results')
    parser.add_argument('--data_info', type=str, default=r'Z:\rubenvdw\Info_files_Dataset_split\15_classes_dataset_split_extraframes_25102024')
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--lipid_cap_area_manual', type=int, default=0)
    parser.add_argument('--task_name', type=str, default=r'Dataset_dummy')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    counts = Get_Distributions(args.data_paths, args.output_filename, args.data_info, args.num_classes, args.lipid_cap_area_manual, args.task_name)
    counts.get_counts()

if __name__ == "__main__":
    main(sys.argv)
