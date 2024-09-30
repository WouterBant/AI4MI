import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import argparse
from skimage.io import imsave

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import RndTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from tqdm import tqdm


# ------------------- Data Loading Component for CT Data -------------------

class CTImageDataset(SlimDataLoaderBase):
    """
    Custom DataLoader to handle paired CT images and ground truth masks for medical image processing.
    """

    def __init__(self, images, gts, batch_size, batch):
        super(CTImageDataset, self).__init__(None, batch_size)
        self.images = [self.add_channel_dim(img) for img in images]  # Add channel dimension
        self.gts = [self.add_channel_dim(gt) for gt in gts]  # Add channel dimension to GT
        self.images = [img.astype(np.float32) for img in self.images]  # Ensure float32 for consistency
        self.gts = [gt.astype(np.float32) for gt in self.gts]  # Ensure float32 for GT as well
        self.batch = batch

    def add_channel_dim(self, img):
        return np.expand_dims(img, axis=0)  # Add channel dimension (1, H, W)

    def generate_train_batch(self):
        # Randomly select indices for the batch
        if self.batch == True:
            indices = np.random.choice(len(self.images), self.batch_size, replace=False)
            batch_data = np.stack([self.images[i] for i in indices], axis=0)
            batch_gts = np.stack([self.gts[i] for i in indices], axis=0)
        else:
            batch_data = np.stack([img for img in self.images], axis=0)
            batch_gts = np.stack([gt for gt in self.gts], axis=0)
        return {'data': batch_data, 'gt': batch_gts}  # Return both input data and GT


def load_ct_images_and_gts(image_folder):
    """
    Load paired CT images and corresponding GT masks from the specified folders.
    """
    images = []
    gts = []
    gt_folder = os.path.join(image_folder, 'gt')
    image_folder = os.path.join(image_folder, 'img')
    print(image_folder)
    print(gt_folder)

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        gt_path = os.path.join(gt_folder, filename)
        
        if os.path.exists(gt_path):
            img = imread(img_path)
            gt = imread(gt_path)
            if img is not None and gt is not None:
                images.append(img)
                gts.append(gt)
    return images, gts


def plot_batch(batch):
    """
    Visualize a batch of images and their ground truth.
    """
    batch_size = batch['data'].shape[0]
    plt.figure(figsize=(16, 10))
    for i in range(batch_size):
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(batch['data'][i, 0], cmap="gray")  # Visualize input image
        plt.axis('off')
        plt.subplot(2, batch_size, batch_size + i + 1)
        plt.imshow(batch['gt'][i, 0], cmap="gray")  # Visualize GT
        plt.axis('off')
    plt.show()


def save_augmented_images(batch, output_folder, prefix="augmented"):
    """
    Save the augmented images and GT masks to the specified output folder.
    
    Args:
        batch (dict): The augmented batch of images and GTs.
        output_folder (str): The directory where the images will be saved.
        prefix (str): The prefix to add to each saved image filename.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (img, gt) in enumerate(tqdm(zip(batch['data'], batch['gt']), total=len(batch['data']), desc="Saving augmented images")):
        img = np.squeeze(img).astype(np.uint8)  # Remove single-dimensional entries from the shape (H, W)
        gt = np.squeeze(gt).astype(np.uint8)  # Remove single-dimensional entries from the shape (H, W)

        img_save_path = os.path.join(output_folder, f"{prefix}_image_{i}.png")
        gt_save_path = os.path.join(output_folder, f"{prefix}_gt_{i}.png")

        imsave(img_save_path, img)
        imsave(gt_save_path, gt)

        print(f"Saved {img_save_path} and {gt_save_path}")


# ------------------- Augmentation Component -------------------

class AugmentationPipeline:
    """
    Defines the augmentation pipeline, which includes spatial transformations, noise, and contrast/brightness augmentations.
    """

    def __init__(self, target_patch_size, oversized_factor=1.2):
        self.target_patch_size = target_patch_size
        self.oversized_patch_size = (int(target_patch_size[0] * oversized_factor), int(target_patch_size[1] * oversized_factor))

        
    def get_rotation_transform(self):
        return RndTransform(SpatialTransform(
            patch_size=self.oversized_patch_size,
            patch_center_dist_from_border=np.array(self.oversized_patch_size) // 2,
            do_rotation=True,
            angle_x=(-0.174532925, 0.174532925),  # Rotate between -10 to +10 degrees on the z-axis

            # Disable all other transformations
            do_elastic_deform=False,  # Disable elastic deformation
            do_scale=False,  # Disable scaling
            random_crop=False,  # Disable random cropping (center crop is used by default)
            independent_scale_for_each_axis=False,  # Disable independent scaling

            # Border handling (to avoid artifacts when rotating)
            border_mode_data='constant',  # Handle borders by padding with constant values
            border_cval_data=0,  # Pad the image with 0 (black) outside its original boundaries
            order_data=1  # Linear interpolation for smooth rotation
        ), prob=0.2)
    
    def get_scale_transform(self):
        return RndTransform(SpatialTransform(
            patch_size=self.oversized_patch_size,  # Output image patch size after padding
            patch_center_dist_from_border=np.array(self.oversized_patch_size) // 2,  # Center the patch

            # Enable rotation
            do_rotation=False,
            # Disable all other transformations
            do_elastic_deform=False,  # Disable elastic deformation
            do_scale=True,  # Disable scaling
            scale=(0.7, 1.4),  # Scale between 0.5 and
            random_crop=False,  # Disable random cropping (center crop is used by default)
            independent_scale_for_each_axis=True,  # Disable independent scaling

            # Border handling (to avoid artifacts when rotating)
            border_mode_data='constant',  # Handle borders by padding with constant values
            border_cval_data=0,  # Pad the image with 0 (black) outside its original boundaries
            order_data=1  # Linear interpolation for smooth rotation
        ), prob=0.2)


    def get_noise_transform(self):
        return GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_sample=0.15  # Probability of applying noise
        )

    def get_blur_transform(self):
        return GaussianBlurTransform(
            blur_sigma=(0.5, 1.5),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5
        )

    def get_brightness_transform(self):
        return BrightnessMultiplicativeTransform(
            multiplier_range=(0.7, 1.3),
            p_per_sample=0.15
        )

    def get_contrast_transform(self):
        return ContrastAugmentationTransform(
            contrast_range=(0.65, 1.5),
            p_per_sample=0.15
        )

    # XXX
    def get_low_res_transform(self):
        return SimulateLowResolutionTransform(
            zoom_range=(1, 2),
            per_channel=True,
            p_per_sample=0.25,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3
        )

    def get_gamma_transform(self):
        return GammaTransform(
            gamma_range=(0.7, 1.5),
            invert_image=True,
            per_channel=True,
            retain_stats=True,
            p_per_sample=0.15,
        )

    def get_mirror_transform(self):
        return MirrorTransform(
            axes=(0, 1, 2),
            p_per_sample=0.5
        )

    def create_pipeline(self):
        """
        Create the full augmentation pipeline.
        """
        return Compose([
            self.get_rotation_transform(),
            self.get_scale_transform(),
            self.get_noise_transform(),
            self.get_blur_transform(),
            self.get_brightness_transform(),
            self.get_contrast_transform(),
            self.get_low_res_transform(),
            self.get_gamma_transform(),
            self.get_mirror_transform()
        ])

    def create_data_generator(self, batchgen, num_processes=4, num_cached_per_queue=2):
        """
        Initialize a multi-threaded data generator with the augmentation pipeline.
        """
        augmentation_pipeline = self.create_pipeline()

        # Create the multi-threaded augmenter
        return MultiThreadedAugmenter(
            batchgen,
            augmentation_pipeline,
            num_processes=num_processes,
            num_cached_per_queue=num_cached_per_queue,
            seeds=None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Augmentation Pipeline')
    parser.add_argument('--data_folder', type=str, default='/Users/sachabuijs/Documents/AI4MI/data/SEGTHOR/train', help='Path to the training data folder')
    parser.add_argument('--output_folder', type=str, default='/Users/sachabuijs/Documents/AI4MI/data/SEGTHOR/augmented', help='Path to the output folder')
    parser.add_argument('--batch', type=bool, default=False, help='Do you want batches or to augment all the data at once?')

    args = parser.parse_args()

    # Load the training images from the folder
    train_images, train_gts = load_ct_images_and_gts(args.data_folder)
    print(len(train_images))
    
    # Initialize the data loader with batch size of 4
    batchgen = CTImageDataset(train_images, train_gts, batch_size=4, batch=args.batch)

    # Initialize the augmentation pipeline with the target patch size
    target_patch_size = train_images[0].shape  # Assuming HxW image
    augmentation_pipeline = AugmentationPipeline(target_patch_size)

    # Create the data generator with augmentations
    augmented_data_generator = augmentation_pipeline.create_data_generator(batchgen)

    # Retrieve a batch of augmented images and visualize
    augmented_batch = next(augmented_data_generator)


    save_augmented_images(augmented_batch, args.output_folder)