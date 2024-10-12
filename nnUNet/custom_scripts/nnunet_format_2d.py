import os
import re

def rename_files(directory):
    # Regular expression to capture the current naming structure
    pattern = re.compile(r"(Patient)_(\d{2})_(\d{4})\.png")

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            prefix, patient_id, sequence = match.groups()

            # Use the sequence as the new middle part, formatted as a 3-digit number
            new_middle_part = f"{int(sequence):03d}"

            # The last part is always '0000' (CHANGE DEPENDING ON IMAGES OR MASKS) -> add _0000 for images
            new_filename = f"{prefix}{patient_id}_{new_middle_part}.png"

            # Rename the file
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage
directory_path = "/home/scur2527/ai4mi/data/nnUNet_raw/Dataset014_segthor_images_ordinary_manual_split/labelsTr/"
rename_files(directory_path)
