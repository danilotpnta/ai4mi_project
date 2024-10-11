"""
This scripts adapts the SegTHOR dataset to the nnU-Net format. It creates the necessary directories and copies the data to the appropriate folders.

More information about data conversion can be found at:
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

"""
import os
import shutil
import json

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                os.remove(file_path)
                

segthor_train_dir = '/home/scur2518/segthor/train'  # Path to your SegTHOR train directory
segthor_test_dir = '/home/scur2518/segthor/test'    # Path to your SegTHOR test directory
nnunet_base_dir = '/home/scur2518/segthor_nnunet'  # Path to nnUNet raw data base
task_id = '201'  # Task ID for nnU-Net
task_name = f'Task{task_id}_SegTHOR'

# Remove all .DS_Store files in the dataset
remove_ds_store(segthor_train_dir)
remove_ds_store(segthor_test_dir)

# Create necessary nnU-Net directories
nnunet_task_dir = os.path.join(nnunet_base_dir, 'nnUNet_raw', task_name)
imagesTr_dir = os.path.join(nnunet_task_dir, 'imagesTr')
labelsTr_dir = os.path.join(nnunet_task_dir, 'labelsTr')
imagesTs_dir = os.path.join(nnunet_task_dir, 'imagesTs')

os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)

# Process training data
for patient_id in os.listdir(segthor_train_dir):
    patient_path = os.path.join(segthor_train_dir, patient_id)
    image_file = os.path.join(patient_path, f'{patient_id}.nii.gz')
    label_file = os.path.join(patient_path, 'GT.nii.gz')

    # Copy image and label to nnU-Net folders with appropriate names
    shutil.copy(image_file, os.path.join(imagesTr_dir, f'{patient_id}_0000.nii.gz'))
    shutil.copy(label_file, os.path.join(labelsTr_dir, f'{patient_id}.nii.gz'))

# Process test data
for test_image in os.listdir(segthor_test_dir):
    patient_id = test_image.replace('.nii.gz', '')
    test_image_path = os.path.join(segthor_test_dir, test_image)

    # Copy test image to nnU-Net test folder
    shutil.copy(test_image_path, os.path.join(imagesTs_dir, f'{patient_id}_0000.nii.gz'))

# Create dataset.json
dataset_json = {
    "channel_names": {
        "0": "CT"  
    },
    "labels": {
        "background": 0,
        "heart": 1,
        "aorta": 2,
        "esophagus": 3,
        "trachea": 4
    },
    "numTraining": len(os.listdir(imagesTr_dir)),
    "numTest": len(os.listdir(imagesTs_dir)),
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",  
    "training": [
        {
            "image": f"./imagesTr/{patient_id}_0000.nii.gz",
            "label": f"./labelsTr/{patient_id}.nii.gz"
        } for patient_id in [x.replace('_0000.nii.gz', '') for x in os.listdir(imagesTr_dir)]
    ],
    "test": [
        {
            "image": f"./imagesTs/{patient_id}_0000.nii.gz"
        } for patient_id in [x.replace('_0000.nii.gz', '') for x in os.listdir(imagesTs_dir)]
    ]
}

# Save dataset.json
with open(os.path.join(nnunet_task_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f"SegTHOR dataset successfully converted to nnU-Net format in {nnunet_task_dir}")