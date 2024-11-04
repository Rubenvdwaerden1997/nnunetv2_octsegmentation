from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json, join
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder

def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


def main():
    task_name = r'Dataset601_TS3D3'
    output_base = r'Z:\rubenvdw\nnunetv2\nnUNet\nnunetv2\Data\nnUNet_raw'
    file_ending='.nii'
    RGB=False
    target_base = join(output_base, task_name)
    pseudo_frames = 3 #This is for pseudo 3d, +-1 frame is k=1. For 2D, set this to 0

    channel_names = {}
    index = 0
    #RGB
    if RGB:
        if pseudo_frames == 0:
            # Assign indices starting from 0
            channel_names[index] = 'R'
            index += 1
            channel_names[index] = 'G'
            index += 1
            channel_names[index] = 'B'
        else:
            for i in range(-pseudo_frames, pseudo_frames + 1):
                channel_names[index] = f'R{i}'
                index += 1
                channel_names[index] = f'G{i}'
                index += 1
                channel_names[index] = f'B{i}'
                index += 1
    else:
        if pseudo_frames == 0:
            channel_names[index] = 'image'
        else:
            for i in range(-pseudo_frames, pseudo_frames + 1):
                channel_names[index] = f'image{i}'
                index += 1
    
    labels={'background':0,'lumen':1,'guidewire':2, 'intima':3, 'lipid':4, 'calcium':5, 'media':6, 'catheter':7, 'sidebranch':8, 'rthrombus':9, 'wthrombus':10, 'rupture':11, 'healed plaque':12, 'neovascularization':13}

    train_identifiers = get_identifiers_from_splitted_dataset_folder(join(target_base, 'imagesTr'), file_ending)
    test_identifiers = get_identifiers_from_splitted_dataset_folder(join(target_base, 'imagesTs'), file_ending)

    if file_ending == '.nii' or file_ending == '.nii.gz':
        overwrite_image_reader_writer = 'SimpleITKIO'
    generate_dataset_json(output_folder=target_base,
                          channel_names=channel_names,
                          labels=labels,
                          num_training_cases=len(train_identifiers),
                          file_ending=file_ending,
                          dataset_name=task_name,
                          description='Test case for generating dataset.json file', 
                          overwrite_image_reader_writer=overwrite_image_reader_writer,
                          numTest=len(test_identifiers))
    
if __name__=='__main__':
    main()