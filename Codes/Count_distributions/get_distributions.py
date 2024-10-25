import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import SimpleITK as sitk
import os
import sys
import argparse
import cv2
sys.path.insert(1, '/data/diag/rubenvdw/nnunetv2/nnUNet/Codes/utils')
from postprocessing import create_annotations_lipid, create_annotations_calcium, intima_cap_area_v2

class Get_Distributions:

    def __init__(self, data_path: str, output_filename: str, data_info: str, num_classes:int,lipid_cap_area_manual:int,task_name:str ):
        """_summary_

        Args:
            data_path (str): path to the folder with the segmentations you want to count (e.g labelsTr, labelsTs or preds from you model)
            output_filename (str): path to the Excel file you want to generate with the generated distributions data
            data_info (str): path to the Excel file with the patients data (i.e train_test_split_v2.xlsx)
        """        
        
        self.data_path = data_path
        self.output_filename = output_filename
        self.data_info = pd.read_excel('{}.xlsx'.format(data_info))
        self.num_classes = num_classes
        self.lipid_cap_area_manual=lipid_cap_area_manual
        self.task_name=task_name


    def get_patient_data(self, file: str) -> str:
        """Processes the name of the file so we retrieve the pullback name

        Args:
            file (str): raw filename of prediction

        Returns:
            str: pullback name processed
        """        

        #Obtain format of pullback name (it's different than in the dataset counting)
        filename = file.split('_')[0]
        first_part = filename[:3]
        second_part = filename[3:-4]
        third_part = filename[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Obtain pullback name
        n_pullback = file.split('_')[1]
        pullback_name = self.data_info[(self.data_info['Nº pullback'] == int(n_pullback)) & (self.data_info['Patient'] == patient_name)]['Pullback'].values[0]

        return pullback_name


    def get_counts(self):

        """Creates Excel file with the label count and lipid and calcium measurements 
            for every frame that contains annotations in the specified predicted folder. You can use this function
            either for the original dataset (i.e labelsTs or labelsTr folder) and the predicted segmentations (i.e in your model predictions)
        """
        if self.num_classes==13:
            counts_per_frame = pd.DataFrame(columns = ['Pullback', 'Frame', 'AI_background', 'AI_lumen', 'AI_guidewire', 'AI_intima', 'AI_lipid', 'AI_calcium', 
                                    'AI_media', 'AI_catheter', 'AI_sidebranch', 'AI_rthrombus', 'AI_wthrombus', 'AI_dissection',
                                    'AI_plaque_rupture', 'AI_lipid_arc', 'AI_FCT', 'AI_calcium_depth', 'AI_calcium_arc', 'AI_calcium_thickness','Manual_FC_Area','AI_FC_Area'])
        elif self.num_classes==15:
            counts_per_frame = pd.DataFrame(columns = ['Pullback', 'Frame',  'AI_background', 'AI_lumen', 'AI_guidewire', 'AI_intima', 'AI_lipid', 'AI_calcium', 
                                    'AI_media', 'AI_catheter', 'AI_sidebranch', 'AI_rthrombus', 'AI_wthrombus', 'AI_dissection',
                                    'AI_plaque_rupture','AI_layered','AI_microvessel', 'AI_lipid_arc', 'AI_FCT', 'AI_calcium_depth', 'AI_calcium_arc', 'AI_calcium_thickness','Manual_FC_Area','AI_FC_Area'])
        elif self.num_classes==11:
            counts_per_frame = pd.DataFrame(columns = ['Pullback', 'Frame',  'AI_background', 'AI_lumen', 'AI_guidewire', 'AI_intima', 'AI_lipid', 'AI_calcium', 
                                    'AI_media', 'AI_catheter', 'AI_sidebranch', 'AI_thrombus','AI_plaque_rupture', 'AI_lipid_arc', 'AI_FCT', 'AI_calcium_depth', 'AI_calcium_arc', 'AI_calcium_thickness','Manual_FC_Area','AI_FC_Area'])
        elif self.num_classes==10:
            counts_per_frame = pd.DataFrame(columns = ['Pullback', 'Frame', 'AI_background', 'AI_lumen', 'AI_guidewire', 'AI_intima', 'AI_lipid', 'AI_calcium',                                    
                                                       'AI_media', 'AI_plaque_rupture', 'AI_sidebranch', 'AI_thrombus', 'AI_lipid_arc', 'AI_FCT', 'AI_calcium_depth', 'AI_calcium_arc', 'AI_calcium_thickness','Manual_FC_Area','AI_FC_Area'])

        for file in os.listdir(self.data_path):

            #Check only nifti files
            if file.endswith('nii.gz') or file.endswith('nii'):

                pullback_name = self.get_patient_data(file)
                n_frame = file.split('_')[2][5:]
                n_frame=int(n_frame)
                print('Counting {} ...'.format(file))

                seg_map = sitk.ReadImage(os.path.join(self.data_path, file))
                seg_map_data = sitk.GetArrayFromImage(seg_map)[0]



                #Get count of labels in each frame
                one_hot = np.zeros(self.num_classes)

                unique, _ = np.unique(seg_map_data, return_counts=True)
                unique = unique.astype(int)

                one_hot[[unique[i] for i in range(len(unique))]] = 1

                #Post-processing results
                _, _ , cap_thickness, lipid_arc, _ = create_annotations_lipid(seg_map_data, font = 'cluster')
                _, _ , calcium_depth, calcium_arc, calcium_thickness, _ = create_annotations_calcium(seg_map_data, font = 'cluster')

                #Lipid Cap area calculations
                if self.lipid_cap_area_manual==1:
                    folder_base = '/data/diag/rubenvdw/nnU-net/data-2d/nnUNet_raw_data'
                    folder_label = os.path.join(folder_base, self.task_name, 'labelsTs')
                    manual_map = sitk.ReadImage(os.path.join(folder_label, file))
                    manual_map_data = sitk.GetArrayFromImage(manual_map)[0]
                    _,manual_lipid_cap_area=intima_cap_area_v2(manual_map_data)
                else:
                    manual_lipid_cap_area=0
                    
                _,ai_lipid_cap_area=intima_cap_area_v2(seg_map_data)
                #Create one hot list with all data
                one_hot_list = one_hot.tolist()
                one_hot_list.insert(0, pullback_name)
                one_hot_list.insert(1, n_frame)
                one_hot_list.append(lipid_arc)
                one_hot_list.append(cap_thickness)
                one_hot_list.append(calcium_depth)
                one_hot_list.append(calcium_arc)
                one_hot_list.append(calcium_thickness)
                one_hot_list.append(manual_lipid_cap_area)
                one_hot_list.append(ai_lipid_cap_area)
                new_row=pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)])
                counts_per_frame = pd.concat([counts_per_frame,new_row.to_frame().T], ignore_index=True)
                #counts_per_frame=counts_per_frame.append(pd.Series(one_hot_list, index=counts_per_frame.columns[:len(one_hot_list)]), ignore_index=True)

        counts_per_frame.to_excel('{}.xlsx'.format(self.output_filename))

    def get_class_weights(self):
        """Obtain the class weights for the training. IMPORTANT: data_path must be the path to the labelsTr, otherwise you dont get the true
        distribution of the training set
        """        

        label_counts = np.zeros(self.num_classes)

        for file in os.listdir(self.data_path):

            seg = sitk.ReadImage(os.path.join(self.data_path, file))
            seg_data = sitk.GetArrayFromImage(seg)

            #Count the nº of pixels for a label 
            unique, counts = np.unique(seg_data, return_counts=True)
            label_counts[unique] += counts
                
        #Get class weight in terms of frequency in the dataset
        total_pixels = 704 * 704 * len(os.listdir(self.data_path))
        class_weights = total_pixels / (self.num_classes * label_counts)

        print("Class weights:")
        for label, weight in enumerate(class_weights):
            print(f"Label {label}: {weight}")

    
def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'Z:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Predictions\Dataset905_SegmentOCT3d3\Predicted_files')
    parser.add_argument('--output_filename', type=str, default=r'Z:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Predictions\Dataset905_SegmentOCT3d3\Metrics')
    parser.add_argument('--data_info', type=str, default=r'Z:\rubenvdw\Info_files_Dataset_split\15_classes_dataset_split_extraframes_13062024')
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--lipid_cap_area_manual', type=int, default=0)
    parser.add_argument('--task_name', type=str, default=r'Dataset905_SegmentOCT3d3')
    args, _ = parser.parse_known_args(argv)

    args = parser.parse_args()

    counts = Get_Distributions(args.data_path, args.output_filename, args.data_info,args.num_classes,args.lipid_cap_area_manual,args.task_name)
    counts.get_counts()


if __name__ == "__main__":
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)