import os

# Directory to check
directory = '/home/scur2527/ai4mi/data/nnUNet_raw/Dataset014_segthor_images_ordinary_manual_split/imagesTr'

# List all files in the directory (excluding subdirectories)
file_count = sum([1 for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

print(f"Number of files in the directory: {file_count}")
