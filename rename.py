import os
import shutil

def move_and_rename_files(patient_dir, patient_num, dest_dir):
    """
    Function to rename and move files from patient directory to nnUNet format.
    
    Parameters:
    - patient_dir: Source directory for the patient's files.
    - patient_num: Patient number to use in the new file names.
    - dest_dir: Destination directory to store the reformatted files.
    """
    gt_file = os.path.join(patient_dir, "GT.nii.gz")
    img_file = os.path.join(patient_dir, f"Patient_{patient_num}.nii.gz")
    
    # Rename and move ground truth (GT) file
    new_gt_filename = f"BRATS_{str(patient_num).zfill(3)}.nii.gz"
    shutil.copy(gt_file, os.path.join(dest_dir, "labelsTr", new_gt_filename))
    
    # Rename and move image file for a single modality
    new_img_filename = f"BRATS_{str(patient_num).zfill(3)}_0000.nii.gz"
    shutil.copy(img_file, os.path.join(dest_dir, "imagesTr", new_img_filename))


def process_patient_data(source_dir, dest_dir):
    """
    Processes each patient's data by moving and renaming files from the source 
    directory to the destination directory in nnUNet format.
    
    Parameters:
    - source_dir: The root directory containing patient subdirectories.
    - dest_dir: The destination root directory to store the reformatted files.
    """
    # Create destination folders if they don't exist
    os.makedirs(os.path.join(dest_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "labelsTr"), exist_ok=True)

    # Iterate through each patient directory in the source folder
    for patient_folder in os.listdir(source_dir):
        patient_path = os.path.join(source_dir, patient_folder)
        if os.path.isdir(patient_path) and patient_folder.startswith("Patient_"):
            patient_num = patient_folder.split('_')[1]  # Extract patient number
            move_and_rename_files(patient_path, patient_num, dest_dir)
    
    print("Files have been successfully moved and renamed.")

def main():
    """
    Main function to initiate the processing of patient data.
    """
    # Update the source and destination directories as per your setup
    source_dir = "data/segthor_train/train"  # Replace with your actual path
    dest_dir = "dataset/nnUNet_raw/Dataset001_Seghtor"  # Replace with your actual path
    
    process_patient_data(source_dir, dest_dir)

if __name__ == "__main__":
    main()
