
from typing import Tuple
import numpy as np
from file_and_folder_operations import *


def get_identifiers_from_splitted_files(folder: str):
    #uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    uniques = np.unique([i[:-9] for i in subfiles(folder, suffix='.nii', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii" % i, "label": "./labelsTr/%s.nii" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


def main():
    target_base = 'Z:/rubenvdw/nnU-net/data-2d/nnUNet_raw_data/Task941_Lipidpaper_10_3D3_nifti'
    target_imagesTr = 'Z:/rubenvdw/nnU-net/data-2d/nnUNet_raw_data/Task941_Lipidpaper_10_3D3_nifti/imagesTr'
    target_imagesTs = 'Z:/rubenvdw/nnU-net/data-2d/nnUNet_raw_data/Task941_Lipidpaper_10_3D3_nifti/imagesTs'
    task_name = 'Task941_Lipidpaper_10_3D3_nifti'
    dataset_description = 'Semantic segmentation of intracoronary OCT pullbacks pseudo-3D (k=3) with 10 classes, extra frames 13062024, trying nifti'
    modalities = ('R', 'G', 'B')

    k = 3 #This is for pseudo 3d, +-1 frame is k=1. You can also comment this for 2D
    modalities = []
    for i in range(-k, k+1):
        modalities.append('R{}'.format(i))
        modalities.append('G{}'.format(i))
        modalities.append('B{}'.format(i))
        
    # #15 classes
    # generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=modalities, dataset_description=dataset_description,
    #                     labels={0:'background', 1:'lumen', 2:'guidewire', 3:'intima', 4:'lipid', 5:'calcium', 
    #                     6:'media', 7:'catheter', 8: 'sidebranch', 9:'rthrombus', 10:'wthrombus',11:'dissection',12:'rupture',13:'healed plaque',14:'neovascularization'}, 
    #                     dataset_name=task_name, license='hands off!')

    # #11 classes
    # generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=modalities, dataset_description=dataset_description,
    #                     labels={0:'background', 1:'lumen', 2:'guidewire', 3:'intima', 4:'lipid', 5:'calcium', 
    #                     6:'media', 7:'catheter', 8: 'sidebranch', 9:'thrombus', 10:'rupture'}, 
    #                     dataset_name=task_name, license='hands off!')
    
    #10 classes
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=modalities, dataset_description=dataset_description,
                        labels={0:'background', 1:'lumen', 2:'guidewire', 3:'intima', 4:'lipid', 5:'calcium', 
                        6:'media', 7:'rupture', 8: 'sidebranch', 9:'thrombus'}, 
                        dataset_name=task_name, license='hands off!')

if __name__ == '__main__':
    main()