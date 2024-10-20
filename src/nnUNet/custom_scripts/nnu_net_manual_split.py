import os
import json


def create_splits_json(image_dir, validation_patients, output_path):
    # Initialize splits structure
    splits = {"train": [], "val": []}

    # Iterate over all image files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            # Extract the patient ID (e.g., Patient01) and slice ID (e.g., 000) from the filename
            patient_id = filename.split("_")[0]
            patient_number = int(patient_id.replace("Patient", ""))

            # Add the full image ID (e.g., Patient01_000) to the appropriate set
            image_id = filename.replace(".png", "")  # Remove .png extension

            if patient_number in validation_patients:
                splits["val"].append(image_id)
            else:
                splits["train"].append(image_id)

    # Write the splits to the output JSON file
    with open(output_path, "w") as outfile:
        json.dump([splits], outfile, indent=4)
    print(f"Splits saved to {output_path}")


# Define the directory containing the images
image_dir = "/home/scur2527/ai4mi/data/nnUNet_raw/Dataset014_segthor_images_ordinary_manual_split/labelsTr"

# Define the patient numbers for the validation set
validation_patients = [2, 6, 8, 16, 21, 30, 35, 39]

# Define the output path for the splits_final.json
output_path = "/home/scur2527/ai4mi/data/nnUNet_preprocessed/Dataset014_segthor_images_ordinary_manual_split/splits_final.json"

# Call the function
create_splits_json(image_dir, validation_patients, output_path)
