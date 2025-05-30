import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def data_loader(db_name: str, root: str) -> str:
    """
    Get the data location for the specified database.
    
    Args:
        db_name: Name of the database
        root: Root directory path
        
    Returns:
        data_location: The location of the data for the specified database
    """
    database_paths = {
        'ERM_W': '/ERM_W',
        'Retina': '/Retina_all',
        'erm': '/erm',
        'Vesicles': '/Vesicles_binary',
    }
    
    if db_name in database_paths:
        return root + database_paths[db_name]
    else:
        raise ValueError(f"Invalid database name provided: {db_name}")


def read_folder(db_name, root, mode, use_masks=None):
    """
    Reads folders for images, preprocessed, labels.

    Args:
        db_name: Name of the database
        root: Root directory path
        mode: 'train', 'validate', or 'test'
        use_masks: Flag to indicate whether to include masks
        
    Returns:
        Paths to image directories, and optionally mask directories
    """
    data_location = data_loader(db_name, root)
    
    if mode == 'train':
        print('Loading train data!')
        if use_masks is not None:
            image_dir = data_location + '/train/images/'
            mask_dir = data_location + '/train/masks/'
            label_dir = data_location + '/train/labels/'
            return image_dir, mask_dir, label_dir
        else:
            image_dir = data_location + '/train/images/'
            label_dir = data_location + '/train/labels/'
            return image_dir, label_dir
            
    elif mode == 'validate':
        print('Loading val data!')
        if use_masks is not None:
            image_dir = data_location + '/validate/images/'
            mask_dir = data_location + '/validate/masks/'
            label_dir = data_location + '/validate/labels/'
            return image_dir, mask_dir, label_dir
        else:
            image_dir = data_location + '/validate/images/'
            label_dir = data_location + '/validate/labels/'
            return image_dir, label_dir
            
    elif mode == 'test':
        print('Loading test data!')
        if use_masks is not None:
            image_dir = data_location + '/test/images/'
            mask_dir = data_location + '/test/masks/'
            label_dir = data_location + '/test/labels/'
            return image_dir, mask_dir, label_dir
        else:
            image_dir = data_location + '/test/images/'
            label_dir = data_location + '/test/labels/'
            return image_dir, label_dir
            
    else:
        print('Data location error!')


class ResizeTransform:
    """Transform class for resizing and cropping images while maintaining aspect ratio."""
    
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image, label):
        """Apply transformation to both image and label."""
        if image.size == self.size and label.size == self.size:
            return image, label
        # Resize each image type without changing its channel format
        image = self.resize_image(image)
        label = self.resize_image(label)
        return image, label

    def resize_image(self, img):
        """Process an individual image through the resize pipeline."""
        img = self.resize_and_maintain_aspect_ratio(img, self.size)
        img = self.pad_if_smaller(img, self.size, fill=self.fill)
        img = self.center_crop(img, self.size)
        return img

    def resize_and_maintain_aspect_ratio(self, img, target_size):
        """Resize image while maintaining its aspect ratio."""
        original_width, original_height = img.size
        target_width, target_height = target_size

        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        return img

    def center_crop(self, img, target_size):
        """Crop image to target size from the center."""
        img_width, img_height = img.size
        target_width, target_height = target_size

        crop_width = (img_width - target_width) // 2
        crop_height = (img_height - target_height) // 2

        return img.crop((crop_width, crop_height, crop_width + target_width, crop_height + target_height))

    @staticmethod
    def pad_if_smaller(img, size, fill=0):
        """Pad image if it's smaller than target size."""
        ow, oh = img.size
        padh = size[1] - oh if oh < size[1] else 0
        padw = size[0] - ow if ow < size[0] else 0
        img = ImageOps.expand(img, border=(padw, padh), fill=fill)
        return img


class VesselSegmentationDataset(Dataset):
    """Dataset class for vessel segmentation tasks."""
    
    def __init__(self, db_name, root, mode, transform=None, desired_height=1024, desired_width=1536, use_masks=None):
        """Initialize the dataset."""
        self.use_masks = use_masks
        
        # Load appropriate directories based on use_masks flag
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
                
        # Set up transform
        if transform is None:
            self.transform = ResizeTransform(size=(desired_width, desired_height))
        else:
            self.transform = transform

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.images)

    @staticmethod
    def process_images(image, label):
        """
        Process images and labels to tensors with appropriate normalization.
        
        Args:
            image: PIL image
            label: PIL label mask
            
        Returns:
            Normalized and formatted PyTorch tensors
        """
        # Convert PIL Images to numpy arrays
        image_np = np.array(image, dtype=np.float32)
        label_np = np.array(label, dtype=np.float32)
        
        # Normalize images to [0,1] range
        if image_np.max() > 1.0:
            image_np /= 255.0
            
        # Convert and threshold labels
        if label_np.max() > 1.0:
            label_np /= 255.0
            label_np = np.round(label_np)
        else:
            label_np = np.round(label_np)
            
        # Handle RGB images (3 channels)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.moveaxis(image_np, -1, 0)
            
        # Handle Grayscale images (1 channel)
        if label_np.ndim == 2:
            label_np = np.expand_dims(label_np, 0)
        
        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image_np)
        label_tensor = torch.from_numpy(label_np)
        
        return image_tensor, label_tensor
    
    def plot_image(self, image, label, title='Image and Label'):
        """Plot image and label side by side for visualization."""
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.array(image), cmap='gray')
        ax[0].set_title('Image')
        ax[1].imshow(np.array(label), cmap='gray')
        ax[1].set_title('Label')
        plt.suptitle(title)
        plt.show()

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Load image and label
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx]).convert('L')
        
        # Apply transformation
        image, label = self.transform(image, label)
        
        # Process images (convert to numpy arrays, normalize, convert to tensors)
        image, label = self.process_images(image, label)
        
        # Get filename for reference
        filename = os.path.basename(self.images[idx])

        return image, label, filename