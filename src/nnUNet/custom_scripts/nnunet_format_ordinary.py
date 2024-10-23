import os
import nibabel as nib
import numpy as np
import sys

def map_float_to_int(image_array):
    """
    Maps float pixel values to their corresponding integer labels.

    Args:
        image_array (numpy.ndarray): The input array with float pixel values.

    Returns:
        numpy.ndarray: A new array with pixel values mapped to integers.
    """
    rounded_array = np.rint(image_array).astype(np.int64)
    modified_array = rounded_array.copy()
    modified_array[rounded_array == 0] = 0
    modified_array[rounded_array == 1] = 1
    modified_array[rounded_array == 2] = 2
    modified_array[rounded_array == 3] = 3
    modified_array[rounded_array == 4] = 4
    return modified_array

def process_directory(image_directory):
    """
    Processes all .nii.gz files in the given directory by mapping pixel values.

    Args:
        image_directory (str): The directory containing .nii.gz files.
    """
    for filename in os.listdir(image_directory):
        if filename.endswith(".nii.gz"):
            image_path = os.path.join(image_directory, filename)
            nii_image = nib.load(image_path)
            image_array = nii_image.get_fdata()
            modified_array = map_float_to_int(image_array)
            modified_nii_image = nib.Nifti1Image(modified_array, nii_image.affine, nii_image.header)
            nib.save(modified_nii_image, image_path)

    print(f"Pixel values updated for all .nii.gz files in {image_directory}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 nnu_net_format_ordinary_3d.py <path_to_directory>")
        sys.exit(1)

    directory_path = sys.argv[1]
    process_directory(directory_path)

