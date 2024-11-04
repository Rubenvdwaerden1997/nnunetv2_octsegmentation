import os
import os.path as osp
import numpy as np
import pandas as pd
import random
import json
from glob import glob
from itertools import islice

def load_dataset(file_path):
    """
    Load the dataset from an Excel file.

    Parameters:
    - file_path (str): Path to the Excel file.

    Returns:
    - DataFrame: Loaded dataset.
    """
    return pd.read_excel(file_path)

def count_split_distribution(ids_list, df, classes_to_include):
    """
    Count the distribution of classes in the given ids.

    Parameters:
    - df (DataFrame): The dataset.
    - pb_ids (list): List of pullback ids.
    - classes_to_include (list): List of classes to include in the count.

    Returns:
    - np.array: Percentage distribution of classes.
    """
    class_counts = np.zeros(len(classes_to_include))
    for i in ids_list:
        row = df.loc[df['Pullback'] == i][classes_to_include].values
        class_counts += row[0]
    perc = np.array([c / class_counts[0] for c in class_counts])
    return perc

def demographics_count_split_distribution(ids_list, df, demographics_to_include):
    """
    Count the distribution of demographics in the given ids.

    Parameters:
    - df (DataFrame): The dataset.
    - ids_list (list): List of pullback ids.
    - demographics_to_include (list): List of demographics to include in the count.

    Returns:
    - dict: Distribution of demographics.
    """
    demographics_counts = {demo: 0 for demo in demographics_to_include}
    for i in ids_list:
        row = df.loc[df['Pullback'] == i][demographics_to_include].values[0]
        for j, demo in enumerate(demographics_to_include):
            if row[j]==-99:
                row[j]=0
            elif demo == 'Dem_age':
                demographics_counts[demo] += row[j]
            elif demo == 'Dem_gender': #Female is converted as 0
                demographics_counts[demo] += 1 if row[j] == 1 else 0
            else:
                demographics_counts[demo] += int(row[j])
    
    demographics_counts['Dem_age'] /= len(ids_list)  # Average age 
    for demo in demographics_to_include:
        if demo != 'Dem_age':
            demographics_counts[demo] = (demographics_counts[demo] / len(ids_list))  # Convert to percentage.
    return np.array(list(demographics_counts.values()))

def save_file_train_test_split(train_ids, test_ids, classes_to_include, test_distribution, train_distribution, task_name, demographics_to_include, test_distribution_demographics, train_distribution_demographics):
    """
    Save the train/test split information to a JSON file.

    Parameters:
    - train_ids (list): List of training ids.
    - test_ids (list): List of testing ids.
    - classes_to_include (list): List of classes.
    - test_distribution (list): Distribution of classes in the test set.
    - train_distribution (list): Distribution of classes in the train set.
    - task_name (str): Name of the task.

    Returns:
    - None
    """
    data = {'train_ids': train_ids,
            'test_ids': test_ids,
            'Classes': classes_to_include,
            'Test_distribution_classes': test_distribution,
            'Train_distribution_classes': train_distribution,
            'Demographics': demographics_to_include,
            'Test_distribution_demographics': test_distribution_demographics,
            'Train_distribution_demographics': train_distribution_demographics
            }
        
    output_dir=osp.join(r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info', task_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, 'train_test_split.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def perform_train_test_split(df, test_fraction=0.1, tol=0.03, classes_to_include=None, demographics_to_include=None):
    """
    Perform the split between train and test set, based on a test_fraction, tolerance of difference for each class between train/test set.

    Parameters:
    - df (DataFrame): The dataset.
    - test_fraction (float): Fraction of data to be used as test set.
    - tol (list): percentage to accept the difference between train and test set.
    - classes_to_include (list): List of classes to include in the count.

    Returns:
    - train_ids (list): List of training ids.
    - test_ids (list): List of testing ids.
    """
    ids_list = list(df['Pullback'])
    max_discrepancy = 1
    max_discrepancy_dem = 1
    it = 1
    intersection = np.array(['Dummy1', 'Dummy2'])
    while max_discrepancy > tol or max_discrepancy_dem > 0.2 or len(intersection) > 0:
        test_ids = random.choices(ids_list, k=int(len(ids_list) * test_fraction))
        train_ids = [p for p in ids_list if p not in test_ids]
        dem_train = demographics_count_split_distribution(train_ids, df, demographics_to_include)[1:] # Exclude the age which should be first
        dem_test = demographics_count_split_distribution(test_ids, df, demographics_to_include)[1:]  # Exclude the age which should be first
        perc_train = count_split_distribution(train_ids, df, classes_to_include)
        perc_test = count_split_distribution(test_ids, df, classes_to_include)
        max_discrepancy = np.max(np.abs(perc_train - perc_test))
        max_discrepancy_dem = np.max(np.abs(dem_train - dem_test))
        print(f'iter {it}')
        it += 1

        # check which patients have pullbacks in both splits
        train_patient_ids = ['-'.join(s.split('-')[0:3]) for s in train_ids]
        test_patient_ids = ['-'.join(s.split('-')[0:3]) for s in test_ids]
        intersection = list(set(train_patient_ids) & set(test_patient_ids))

    print(f'max_discrepancy in classes {max_discrepancy} and demographics {max_discrepancy_dem}')
    print(f'for class: {classes_to_include[np.argmax(np.abs(perc_train - perc_test))]}')
    print(f'for demographics: {demographics_to_include[np.argmax(np.abs(dem_train - dem_test))+1]}')

    return train_ids, test_ids

if __name__ == "__main__":
    file_path = r'Z:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Data_info/Dataset_Characteristics_14_classes_29102024.xlsx'
    df = load_dataset(file_path)
    task_name = 'Dummy_task_name_toldem_0.1'
    classes_to_include = ['Background', 'Lumen', 'Guidewire', 'Intima', 'Lipid', 'Calcium', 'Media',
                          'Catheter', 'Sidebranch', 'Red thrombus','White thrombus','Plaque rupture','Layered plaque', 'Neovascularization']
    # I exclude MH_Smoke and Index_Event, because there are 3 categories, and i do not have enough data to make a split among these three groups
    # Dem_age is not being used in the split, but is assumed to be equally distributed among the train and test set
    #demographics_to_include = ['Dem_age','Dem_gender','MH_MI','MH_DM','MH_HT','MH_Smoke','MH_HC','Index_Event','MedPresentation_LipidLowTher']
    demographics_to_include = ['Dem_age','Dem_gender','MH_MI','MH_DM','MH_HT','MH_HC','MedPresentation_LipidLowTher']
    train_ids, test_ids = perform_train_test_split(df, test_fraction=0.1,tol=0.05, classes_to_include=classes_to_include, demographics_to_include=demographics_to_include)
    perc_train = count_split_distribution(train_ids, df, classes_to_include)
    perc_test = count_split_distribution(test_ids, df, classes_to_include)
    perc_dem_train = demographics_count_split_distribution(train_ids, df, demographics_to_include)
    perc_dem_test = demographics_count_split_distribution(test_ids, df, demographics_to_include)
    save_file_train_test_split(train_ids, test_ids, classes_to_include, list(perc_test), list(perc_train), task_name, demographics_to_include, list(perc_dem_test), list(perc_dem_train))