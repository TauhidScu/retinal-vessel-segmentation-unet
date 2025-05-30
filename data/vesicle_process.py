import os
import logging
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def data_loader(db_name: str, root: str) -> str:
    """
    Get the data location for the specified database.
    
    Parameters:
        db_name (str): Name of the database
        root (str): Root directory path
    
    Returns:
        str: The full path to the database
    """
    database_paths = {
        'ERM_W': '/ERM_W',
        'new_erm': '/new_erm',
        'erm': '/erm',
        'DRIVE': '/DRIVE',
        'vesicles': '/vesicles',
    }
    
    if db_name in database_paths:
        return root + database_paths[db_name]
    else:
        raise ValueError(f"Invalid database name provided: {db_name}")


def match_images_and_labels(image_dir, label_dir):
    """
    Match the images and labels by filename and log any mismatches.
    
    Parameters:
        image_dir (str): Directory containing images
        label_dir (str): Directory containing labels
        
    Returns:
        bool: True if all images have matching labels, False otherwise
    """
    # Get filenames without extensions
    image_files = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) 
                         if os.path.isfile(os.path.join(image_dir, f))])
    label_files = sorted([os.path.splitext(f)[0] for f in os.listdir(label_dir) 
                         if os.path.isfile(os.path.join(label_dir, f))])

    # Log counts
    logger.info(f"Total images found: {len(image_files)}")
    logger.info(f"Total labels found: {len(label_files)}")

    # Find mismatches
    missing_labels = []
    missing_images = []

    for image_file in image_files:
        if image_file not in label_files:
            missing_labels.append(image_file)
            logger.warning(f"Missing label for image: {image_file}")

    for label_file in label_files:
        if label_file not in image_files:
            missing_images.append(label_file)
            logger.warning(f"Missing image for label: {label_file}")

    # Log summary
    if missing_labels or missing_images:
        logger.error("Mismatch detected! Check the warnings for details.")
        return False
    else:
        logger.info("All images and labels are correctly matched!")
        return True


def read_folder(db_name, root, mode, use_masks=None):
    """
    Read folders for images, masks (optional), and labels.

    Parameters:
        db_name (str): Name of the database
        root (str): Root directory location
        mode (str): Data split ('train', 'validate', 'test')
        use_masks (bool): Whether to use masks

    Returns:
        tuple: Paths to image, mask (if use_masks), and label directories
    """
    data_location = data_loader(db_name, root)

    if mode in ['train', 'validate', 'test']:
        logger.info(f'Loading {mode} data!')
        
        # Set directory paths based on mode
        base_dir = data_location + f'/{mode}'
        image_dir = base_dir + '/images/'
        label_dir = base_dir + '/labels/'
        
        # Verify images and labels match
        match_images_and_labels(image_dir, label_dir)
        
        # Return with or without masks
        if use_masks is not None:
            mask_dir = base_dir + '/masks/'
            return image_dir, mask_dir, label_dir
        else:
            return image_dir, label_dir
    else:
        logger.error('Invalid mode specified!')
        raise ValueError(f"Invalid mode provided: {mode}")


class ResizeTransform:
    """
    Transform to resize and process image and label pairs consistently.
    """
    def __init__(self, size, fill=0):
        """
        Initialize with target size and fill value for padding.
        """
        self.size = size
        self.fill = fill

    def __call__(self, image, label):
        """
        Apply transformations to both image and label.
        """
        image = self.resize_image(image)
        label = self.resize_image(label)
        return image, label

    def resize_image(self, img):
        """
        Complete resize pipeline for an image.
        """
        img = self.resize_and_maintain_aspect_ratio(img, self.size)
        img = self.pad_if_smaller(img, self.size, fill=self.fill)
        img = self.center_crop(img, self.size)
        return img

    def resize_and_maintain_aspect_ratio(self, img, target_size):
        """
        Resize image while maintaining aspect ratio.
        """
        original_width, original_height = img.size
        target_width, target_height = target_size

        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    def center_crop(self, img, target_size):
        """
        Crop image to target size from the center.
        """
        img_width, img_height = img.size
        target_width, target_height = target_size

        crop_width = (img_width - target_width) // 2
        crop_height = (img_height - target_height) // 2

        return img.crop((crop_width, crop_height, 
                        crop_width + target_width, 
                        crop_height + target_height))

    @staticmethod
    def pad_if_smaller(img, size, fill=0):
        """
        Pad image if smaller than target size.
        """
        ow, oh = img.size
        padh = size[1] - oh if oh < size[1] else 0
        padw = size[0] - ow if ow < size[0] else 0
        img = ImageOps.expand(img, border=(padw, padh), fill=fill)
        return img


class VesicleSegmentationDataset(Dataset):
    """
    Dataset for vesicle segmentation with support for masks.
    """
    def __init__(self, db_name, root, mode, transform=None, 
                 desired_height=1024, desired_width=1536, use_masks=None):
        """
        Initialize dataset with paths and transforms.
        """
        self.use_masks = use_masks
        self.mode = mode
        
        # Load file paths based on whether masks are used
        if self.use_masks is not None:
            images_loc, masks_loc, labels_loc = read_folder(db_name, root, mode, self.use_masks)
            self.images = [os.path.join(images_loc, f) for f in sorted(os.listdir(images_loc)) 
                          if os.path.isfile(os.path.join(images_loc, f))]
            self.masks = [os.path.join(masks_loc, f) for f in sorted(os.listdir(masks_loc)) 
                         if os.path.isfile(os.path.join(masks_loc, f))]
            self.labels = [os.path.join(labels_loc, f) for f in sorted(os.listdir(labels_loc)) 
                          if os.path.isfile(os.path.join(labels_loc, f))]
        else:
            images_loc, labels_loc = read_folder(db_name, root, mode)
            self.images = [os.path.join(images_loc, f) for f in sorted(os.listdir(images_loc)) 
                          if os.path.isfile(os.path.join(images_loc, f))]
            self.labels = [os.path.join(labels_loc, f) for f in sorted(os.listdir(labels_loc)) 
                          if os.path.isfile(os.path.join(labels_loc, f))]

        # Verify matching counts
        if len(self.images) != len(self.labels):
            logger.error(f"Number of images: {len(self.images)} does not match number of labels: {len(self.labels)}")
        else:
            logger.info(f"Number of images: {len(self.images)} matches number of labels: {len(self.labels)}")
        
        # Set transform
        if transform is None:
            self.transform = ResizeTransform(size=(desired_width, desired_height))
        else:
            self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images)

    @staticmethod
    def process_images(image, label):
        """
        Process images and labels for model input.
        
        Converts PIL images to tensors with proper normalization and dimensions.
        """
        # Convert PIL Images to numpy arrays
        image_np = np.array(image, dtype=np.float32)
        label_np = np.array(label, dtype=np.int64)  # Use int64 for multi-class labels

        # Normalize the images to [0, 1]
        image_np /= 255.0

        # Handle RGB images (3 channels), convert from HWC to CHW format
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.moveaxis(image_np, -1, 0)  # Moves channels to first dimension

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image_np)
        label_tensor = torch.from_numpy(label_np)

        return image_tensor, label_tensor
    
    def plot_image(self, image, label, title='Image and Label'):
        """
        Plot image and label side by side for visualization.
        """
        # Convert tensors to numpy arrays
        if image.ndimension() == 3:  # Check if it's a 3-channel image
            image = image.numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        else:  # Grayscale image
            image = image.numpy().squeeze()
            
        label = label.numpy().squeeze()

        # Create plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        ax[0].set_title('Image')
        ax[1].imshow(label, cmap='gray')
        ax[1].set_title('Label')
        plt.suptitle(title)
        plt.show()

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        # Load images
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        label = label.convert('L')  # Convert label to grayscale
        
        # Apply transformations
        image, label = self.transform(image, label)
        
        # Process images
        image, label = self.process_images(image, label)
        
        # For test mode, return filename too
        if self.mode == 'test':
            filename = os.path.basename(self.images[idx])
            return image, label, filename

        return image, label
