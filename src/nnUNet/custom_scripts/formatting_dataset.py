import os
import shutil

# Define paths
input_dir = "../data/segthor_train/train"
output_dir = "../data/nnUNet_raw/Dataset011_segthor"

# Create nnU-Net directories
os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

# Process patients
for patient_folder in os.listdir(input_dir):
    patient_path = os.path.join(input_dir, patient_folder)

    if os.path.isdir(patient_path):
        # Move image file to imagesTr and rename
        img_src = os.path.join(patient_path, f"{patient_folder}.nii.gz")
        img_dst = os.path.join(output_dir, "imagesTr", f"{patient_folder}_0000.nii.gz")
        shutil.copy(img_src, img_dst)

        # Move GT file to labelsTr and rename
        gt_src = os.path.join(patient_path, "GT.nii.gz")
        gt_dst = os.path.join(output_dir, "labelsTr", f"{patient_folder}.nii.gz")
        shutil.copy(gt_src, gt_dst)

print("Dataset restructuring complete!")
