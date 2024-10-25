import SimpleITK as sitk
import os
import json
import numpy as np
import pandas as pd
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
import argparse
sys.path.insert(1, '/data/diag/rubenvdw/nnunetv2/nnUNet/Codes/utils')
from counts_utils import merge_frames_into_pullbacks
from postprocessing import create_annotations_lipid, create_annotations_calcium, compute_arc_dices
from metrics_utils import mean_metrics, calculate_confusion_matrix, metrics_from_cm


class Metrics:

    def __init__(self, orig_folder: str, preds_folder:str, data_info:str, filename:str,num_classes:int,output_folder:str,counts_testset:str,counts_predictiontestset:str):
        """Class to get all the metrics files for a specific model
        Args:
            orig_folder (str): path to your original labels folder (just the raw_data -> TaskXXX -> labelsTs)
            preds_folder (str): path to your preds folder
            data_info (str): path to the Excel file with the patients data (i.e train_test_split_v2.xlsx)
            filename (str): arbitrary ID for the filename to save the Excel file
        """        

        self.orig_folder                = orig_folder
        self.preds_folder               = preds_folder
        self.data_info                  = pd.read_excel(data_info)
        self.filename                   = filename     
        self.num_classes                = num_classes
        self.output_folder              = output_folder
        self.counts_testset             = pd.read_excel(counts_testset)
        self.counts_predictiontestset   = pd.read_excel(counts_predictiontestset)

    def get_all_metrics(self):
        # Create an Excel writer object
        excel_writer = pd.ExcelWriter("{}/{}_all_metrics.xlsx".format(self.output_folder,self.filename))

        # Call all metrics functions and store their results in separate sheets
        dice_per_frame_df = self.dice_per_frame(excel_writer)
        self.TP_dice_per_frame(excel_writer, dice_per_frame_df)
        #self.dice_per_pullback(excel_writer)
        self.get_other_metrics_detection(excel_writer)
        self.get_arc_dice_per_frame(excel_writer)
        #self.get_arc_dice_per_pullback(excel_writer)



        # Save the Excel file
        excel_writer.close()
        print('Done! Find all metrics in {}/{}_all_metrics.xlsx'.format(self.output_folder,self.filename))

    def translate_label(self,label):
        """Translate numerical label to name using label_names dictionary"""
        if self.num_classes==15:
            label_names = {
                0.0: 'background',
                1.0: 'lumen',
                2.0: 'guidewire',
                3.0: 'intima',
                4.0: 'lipid',
                5.0: 'calcium',
                6.0: 'media',
                7.0: 'catheter',
                8.0: 'sidebranch',
                9.0: 'rthrombus',
                10.0: 'wthrombus',
                11.0: 'dissection',
                12.0: 'plaque_rupture',
                13.0: 'layered',
                14.0: 'microvessel',
            }
        elif self.num_classes==11:
            label_names = {
                0.0: 'background',
                1.0: 'lumen',
                2.0: 'guidewire',
                3.0: 'intima',
                4.0: 'lipid',
                5.0: 'calcium',
                6.0: 'media',
                7.0: 'catheter',
                8.0: 'sidebranch',
                9.0: 'thrombus',
                10.0: 'plaque_rupture',
            }
        elif self.num_classes==10:
            label_names = {
                0.0: 'background',
                1.0: 'lumen',
                2.0: 'guidewire',
                3.0: 'intima',
                4.0: 'lipid',
                5.0: 'calcium',
                6.0: 'media',
                7.0: 'plaque_rupture',
                8.0: 'sidebranch',
                9.0: 'thrombus',
            }
        return label_names.get(float(label), f'Unknown-{label}')


    def get_patient_data(self, file: str) -> str:
        """Processes the name of the file so we retrieve the pullback name
        Args:
            file (str): raw filename of prediction
        Returns:
            str: pullback name processed
        """        

        #Get patient name
        patient_name_raw = file.split('_')[0]
        first_part = patient_name_raw[:3]
        second_part = patient_name_raw[3:-4]
        third_part = patient_name_raw[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Get pullback_name
        n_pullback = file.split('_')[1]
        pullback_name = self.data_info[(self.data_info['Nº pullback'] == int(n_pullback)) & (self.data_info['Patient'] == patient_name)]['Pullback'].values[0]

        return pullback_name
    
    def dice_per_frame(self, excel_writer):
        """Obtain the DICE per frame, which are stored in a DataFrame and then to an Excel sheet
        Args:
            excel_writer: Excel writer object to write the DataFrame
        """        

        print('Getting DICE per frame...')
        json_results_file = os.path.join(self.preds_folder, 'summary.json')

        # Load summary file generated by nnUnet
        with open(json_results_file) as f:
            summary = json.load(f)

        final_data = []

        for file in os.listdir(os.path.join(self.preds_folder)):
            if file.endswith('.nii.gz') or file.endswith('.nii'):

                list_dicts_per_frame = []

                # Get pullback name
                pullback_name = self.get_patient_data(file)
                frame = file.split('_')[2][5:]
                # Get DICE score from frame by looking at the json file
                for sub_dict in summary['metric_per_case']:                    
                    normalized_pred_file = sub_dict['prediction_file'].replace('\\', '/')
                    if normalized_pred_file == os.path.join(self.preds_folder, file):
                        list_dicts_per_frame.append({self.translate_label(k): v for i, (k, v) in enumerate(sub_dict['metrics'].items())})

                # Calculate mean DICE scores
                mean_result = mean_metrics(list_dicts_per_frame)

                # Include frame and pullback
                mean_result['Pullback'] = pullback_name
                mean_result['Frame'] = frame

                final_data.append(mean_result)

        # Convert the list of dictionaries to DataFrame
        df = pd.DataFrame(final_data)

        # Write DataFrame to Excel sheet
        df.to_excel(excel_writer, sheet_name="DICE_per_frame", index=False)

        print('Done writing DICE per frame to Excel sheet.')

        return df

    def TP_dice_per_frame(self, excel_writer, dice_per_frame_df):
        # Initialize an empty list to store the final data
        final_data = []
        start_index = list(self.counts_predictiontestset.columns).index('AI_lumen') 
        end_index = list(self.counts_predictiontestset.columns).index('AI_lipid_arc') 
        self.counts_testset['Frame']=self.counts_testset['Frame'].astype(int)
        self.counts_predictiontestset['Frame']=self.counts_predictiontestset['Frame'].astype(int)
        # Iterate over the rows of the dice_per_frame_df DataFrame
        for i in range(len(dice_per_frame_df)):
            # Initialize an empty dictionary to store the data for this row
            row_data = {}

            # Get the pullback name and the frame from the current row in dice_per_frame_df
            pullback = dice_per_frame_df.loc[i, 'Pullback']
            frame = dice_per_frame_df.loc[i, 'Frame']
            print(f'True positive Dice of pullback: {pullback}')
            # Add the pullback name and the frame to the row data
            row_data['Pullback'] = pullback
            row_data['Frame'] = frame
            # Find the row in counts_testset and counts_predictiontestset where 'Pullback' and 'Frame' match the current row in dice_per_frame_df
            matching_row_testset = self.counts_testset.loc[(self.counts_testset['Pullback'] == pullback) & (self.counts_testset['Frame'] == int(frame))]
            matching_row_predictiontestset = self.counts_predictiontestset.loc[(self.counts_predictiontestset['Pullback'] == pullback) & (self.counts_predictiontestset['Frame'] == int(frame))]
            # Iterate over the columns of the counts DataFrames
            for column in self.counts_predictiontestset.columns[start_index:end_index]:
                column_pred=column
                column_test=column
                column_TPdice=column[3:]
                # Check if both counts have a 1 for this class
                if int(matching_row_testset[column_test].values[0]) == 1 and int(matching_row_predictiontestset[column_pred].values[0]) == 1:
                    dice_score = dice_per_frame_df.loc[i, column_TPdice]
                    # Add the Dice score to the row data
                    row_data[column_TPdice] = dice_score
                else:
                    # Add NaN to the row data
                    row_data[column_TPdice] = np.nan
            # Add the row data to the final data
            final_data.append(row_data)
        # Convert the final data to a DataFrame
        df = pd.DataFrame(final_data)

        # Write the DataFrame to an Excel sheet
        df.to_excel(excel_writer, sheet_name="TP_DICE_per_frame", index=False)

    def dice_per_pullback(self, excel_writer):
        """Obtain the DICE per pullback, which are stored in a DataFrame and then to an Excel sheet

        Args:
            excel_writer: Excel writer object to write the DataFrame
        """        

        print('Getting DICE per pullback...')

        # Get frames that belong to each pullback
        merged_pullbacks = merge_frames_into_pullbacks(self.preds_folder)

        final_data = []

        for pullback in merged_pullbacks.keys():

            pullback_name = self.get_patient_data(pullback)

            print('Pullback ', pullback)

            dices_dict = {}

            cm_total = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

            for frame in merged_pullbacks[pullback]:
                # Load original and pred segmentation
                seg_map_data_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(os.path.join(self.preds_folder), frame)))[0]
                seg_map_data_orig = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.orig_folder, frame)))[0]

                # Sum cm for each frame so at the end we get the CM for the whole pullback
                cm = calculate_confusion_matrix(seg_map_data_orig, seg_map_data_pred, range(self.num_classes))       
                cm_total += cm

            dice, _, _, _, _, _ = metrics_from_cm(cm_total)

            # Translate label numbers to label names
            translated_dice_dict = {self.translate_label(label): dice_value for label, dice_value in enumerate(dice)}

            final_data.append({'Pullback': pullback_name, **translated_dice_dict})

        # Convert the list of dictionaries to DataFrame
        df = pd.DataFrame(final_data)

        # Write DataFrame to Excel sheet
        df.to_excel(excel_writer, sheet_name="DICE_per_pullback", index=False)

        print('Done writing DICE per pullback to Excel sheet.')


    def get_other_metrics_detection(self, excel_writer):
        """Obtain PPV, NPV, sensitivity, specificity, and Cohen's kappa based on labels that appear in each frame, 
        which are stored in a DataFrame and then to an Excel sheet
        Args:
            excel_writer: Excel writer object to write the DataFrame
        """    

        print('Getting other metrics for detection...')
        cm_dict = {}

        for file in os.listdir(self.preds_folder):

            if file.endswith('nii.gz'):
            
                print('Checking case', file[:-7])
                orig_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.orig_folder, file)))
                pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.preds_folder, file)))

                # Get labels that appear in frame
                unique_orig = np.unique(orig_seg)
                unique_pred = np.unique(pred_seg)

                # First, we want to check which labels occur and do not occur in both prediction and original
                for label in range(1, self.num_classes):

                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0
                    
                    if label in unique_orig and label in unique_pred:
                        tp += 1

                    elif label not in unique_orig and label not in unique_pred:
                        tn += 1

                    elif label not in unique_orig and label in unique_pred:
                        fp += 1

                    elif label in unique_orig and label not in unique_pred:
                        fn += 1

                    # Create dictionary with the CM values for every label
                    if label not in cm_dict:
                        cm_dict[label] = [tp, tn, fp, fn]

                    else:
                        cm_dict[label][0] += tp
                        cm_dict[label][1] += tn
                        cm_dict[label][2] += fp
                        cm_dict[label][3] += fn

        # Create new dict with metrics using the CM dict
        final_data = {'Label':[], 'PPV': [], 'NPV': [], 'Sensitivity': [], 'Specificity': [], 'Kappa': []}
        for label in cm_dict.keys():
            tp_total, tn_total, fp_total, fn_total = cm_dict[label]

            label_name = self.translate_label(label)

            # Check when there are no TP, FP, FN or TN
            try:
                ppv = tp_total / (tp_total + fp_total)  # precision
            except ZeroDivisionError:
                ppv = 'NaN'

            try:
                npv = tn_total / (tn_total + fn_total)  
            except ZeroDivisionError:
                npv = 'NaN'

            try:
                sens = tp_total / (tp_total + fn_total) # recall
            except ZeroDivisionError:
                sens = 'NaN'

            try:
                spec = tn_total / (tn_total + fp_total)
            except ZeroDivisionError:
                spec = 'NaN'

            try:
                kappa =  2 * (tp_total*tn_total - fn_total*fp_total) / float((tp_total+fp_total)*(fp_total+tn_total) + (tp_total+fn_total)*(fn_total+tn_total))
            except ZeroDivisionError:
                kappa = 'NaN'

            final_data['Label'].append(label_name)
            final_data['PPV'].append(ppv)
            final_data['NPV'].append(npv)
            final_data['Sensitivity'].append(sens)
            final_data['Specificity'].append(spec)
            final_data['Kappa'].append(kappa)

        # Convert the dict to DataFrame
        df = pd.DataFrame(final_data)

        # Write DataFrame to Excel sheet
        df.to_excel(excel_writer, sheet_name="Other_metrics_detection", index=False)

        print('Done writing other metrics for detection to Excel sheet.')


    def get_arc_dice_per_frame(self, excel_writer):
        """Obtain the DICE score for lipid and calcium arcs per frame, which are stored in a DataFrame and then to an Excel sheet

        Args:
            excel_writer: Excel writer object to write the DataFrame
        """        

        print('Getting lipid and calcium arc DICE per frame...')

        orig_segs = os.listdir(self.orig_folder)

        final_data = []

        for seg in orig_segs:

            print('Case ', seg[:-7])

            # Obtain format of pullback name as in the beginning
            pullback_name = self.get_patient_data(seg)

            # Obtain nº frame and set (train/test)
            n_frame = seg.split('_')[2][5:]

            # Read original and pred segmentation
            orig_img = sitk.ReadImage(os.path.join(self.orig_folder, seg))
            orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

            pred_img = sitk.ReadImage(os.path.join(self.preds_folder, seg))
            pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

            # Get IDs for both cases
            _, _, _, _, orig_ids_lipid = create_annotations_lipid(orig_img_data, font='cluster')
            _, _, _, _, pred_ids_lipid = create_annotations_lipid(pred_img_data, font='cluster')

            _, _, _, _, _, orig_ids_calcium = create_annotations_calcium(orig_img_data, font='cluster')
            _, _, _, _, _, pred_ids_calcium = create_annotations_calcium(pred_img_data, font='cluster')

            # Compute new DICE for lipid
            dice_score_lipid, _, _, _ = compute_arc_dices(orig_ids_lipid, pred_ids_lipid)
            dice_score_calcium, _, _, _ = compute_arc_dices(orig_ids_calcium, pred_ids_calcium)

            final_data.append({'pullback': pullback_name, 'frame': n_frame, 'DICE lipid': dice_score_lipid, 'DICE calcium': dice_score_calcium})

        # Convert the list of dictionaries to DataFrame
        df = pd.DataFrame(final_data)

        # Write DataFrame to Excel sheet
        df.to_excel(excel_writer, sheet_name="Arc_DICE_per_frame", index=False)

        print('Done writing arc DICE per frame to Excel sheet.')

    def get_arc_dice_per_pullback(self, excel_writer):
        """Obtain the DICE score for lipid arc and calcium arc on pullback level, which are stored in a DataFrame and then to an Excel sheet

        Args:
            excel_writer: Excel writer object to write the DataFrame
        """      

        print('Getting lipid and calcium arc DICE per pullback...')
        pullback_dict = merge_frames_into_pullbacks(self.preds_folder)

        final_data = []

        for pullback in pullback_dict.keys():

            print('Pullback ', pullback)

            # In order to get DICEs pullback-level, we obtain all of the bin IDs for lipid in every frame with annotation in a pullback

            # Obtain pullback name
            pullback_name = self.get_patient_data(pullback)

            tp_total_lipid = 0
            fp_total_lipid = 0
            fn_total_lipid = 0

            tp_total_calcium = 0
            fp_total_calcium = 0
            fn_total_calcium = 0

            for frame in pullback_dict[pullback]:

                print('Checking frame ', frame)

                orig_img = sitk.ReadImage(os.path.join(self.orig_folder, frame))
                orig_img_data = sitk.GetArrayFromImage(orig_img)[0]

                pred_img = sitk.ReadImage(os.path.join(self.preds_folder, frame))
                pred_img_data = sitk.GetArrayFromImage(pred_img)[0]

                # Get IDs for both cases
                _, _, _, _, orig_ids_lipid = create_annotations_lipid(orig_img_data, font='cluster')
                _, _, _, _, pred_ids_lipid = create_annotations_lipid(pred_img_data, font='cluster')

                _, _, _, _, _, orig_ids_calcium = create_annotations_calcium(orig_img_data, font='cluster')
                _, _, _, _, _, pred_ids_calcium = create_annotations_calcium(pred_img_data, font='cluster')

                # Sum all the TP, TN, FP, FN over a full pullback to get the DICE per pullback
                _, tp_lipid, fp_lipid, fn_lipid = compute_arc_dices(orig_ids_lipid, pred_ids_lipid)
                tp_total_lipid += tp_lipid
                fp_total_lipid += fp_lipid
                fn_total_lipid += fn_lipid

                _, tp_calcium, fp_calcium, fn_calcium = compute_arc_dices(orig_ids_calcium, pred_ids_calcium)
                tp_total_calcium += tp_calcium
                fp_total_calcium += fp_calcium
                fn_total_calcium += fn_calcium

            # Compute new DICE for lipids in a pullback
            try:
                dice_score_pullback_lipid = 2*tp_total_lipid / (tp_total_lipid + fp_total_lipid + tp_total_lipid + fn_total_lipid)

            except ZeroDivisionError:
                dice_score_pullback_lipid = np.nan 

            try:
                dice_score_pullback_calcium = 2*tp_total_calcium / (tp_total_calcium + fp_total_calcium + tp_total_calcium + fn_total_calcium)

            except ZeroDivisionError:
                dice_score_pullback_calcium = np.nan 

            final_data.append({'pullback': pullback_name, 'DICE lipid': dice_score_pullback_lipid, 'DICE calcium': dice_score_pullback_calcium})

        # Convert the list of dictionaries to DataFrame
        df = pd.DataFrame(final_data)

        # Write DataFrame to Excel sheet
        df.to_excel(excel_writer, sheet_name="Arc_DICE_per_pullback", index=False)

        print('Done writing arc DICE per pullback to Excel sheet.')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_folder', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data/nnUNet_raw/Dataset905_SegmentOCT3d3/labelsTs')
    parser.add_argument('--preds_folder', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Predicted_files')
    parser.add_argument('--data_info', type=str, default=r'Z:/rubenvdw/Info_files_Dataset_split/15_classes_dataset_split_extraframes_13062024.xlsx')
    parser.add_argument('--filename', type=str, default='Dataset905_SegmentOCT3d3')
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--output_folder', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Metrics')
    parser.add_argument('--counts_testset', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Metrics/Counts_predictions.xlsx')
    parser.add_argument('--counts_predictiontestset', type=str, default=r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Predictions/Dataset905_SegmentOCT3d3/Metrics/Counts_labels.xlsx')
    args, _ = parser.parse_known_args(argv)
    

    metrics = Metrics(args.orig_folder, args.preds_folder, args.data_info, args.filename, args.num_classes,args.output_folder, args.counts_testset, args.counts_predictiontestset)
    metrics.get_all_metrics()

if __name__ == "__main__":
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)
