import os
from PIL import Image
import numpy as np

# Function to map the pixel values
def map_pixel_values(image_array):
    # Create a copy of the array to modify pixel values
    modified_array = image_array.copy()

    # Mapping the specified pixel values
    modified_array[image_array == 63] = 1
    modified_array[image_array == 126] = 2
    modified_array[image_array == 189] = 3
    modified_array[image_array == 252] = 4
    
    return modified_array

# Directory containing the PNG images
image_directory = '/home/scur2527/ai4mi/data/nnUNet_raw/Dataset014_segthor_images_ordinary_manual_split/labelsTr'

# Iterate through all the files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".png"):
        image_path = os.path.join(image_directory, filename)

        # Load the image
        image = Image.open(image_path)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Modify the pixel values
        modified_array = map_pixel_values(image_array)

        # Convert the NumPy array back to an image
        modified_image = Image.fromarray(modified_array)

        # Save the modified image (overwrite the original or create a new one)
        modified_image.save(image_path)  # Overwrites the original image

print("Pixel values updated for all images in the directory.")
