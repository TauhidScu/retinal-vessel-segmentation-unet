import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def data_loader(db_name: str, root: str) -> str:
    """
    Get the data directory path for the specified database.
    
    Args:
        db_name: Name of the database
        root: Root directory path
    
    Returns:
        The full path to the database directory
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
    Read data directories for images, masks, and labels.
    
    Args:
        db_name: Name of the database
        root: Root directory path
        mode: 'train', 'validate', or 'test'
        use_masks: Whether to use masks or not
        
    Returns:
        Paths to image, mask (optional), and label directories
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
    """Transform class for resizing and processing images while maintaining aspect ratio."""
    
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image, label):
        """Apply transformations to both image and label."""
        if image.size == self.size and label.size == self.size:
            return image, label
        
        image = self.resize_image(image)
        label = self.resize_image(label)
        return image, label

    def resize_image(self, img):
        """Apply the complete resize pipeline to an image."""
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
        """Crop the image from the center to the target size."""
        img_width, img_height = img.size
        target_width, target_height = target_size

        crop_width = (img_width - target_width) // 2
        crop_height = (img_height - target_height) // 2

        return img.crop((crop_width, crop_height, crop_width + target_width, crop_height + target_height))

    @staticmethod
    def pad_if_smaller(img, size, fill=0):
        """Pad the image if it's smaller than the target size."""
        ow, oh = img.size
        padh = size[1] - oh if oh < size[1] else 0
        padw = size[0] - ow if ow < size[0] else 0
        img = ImageOps.expand(img, border=(padw, padh), fill=fill)
        return img


class VesselSegmentationDataset(Dataset):
    """Dataset class for vessel segmentation tasks."""
    
    def __init__(self, db_name, root, mode, transform=None, desired_height=1024, desired_width=1536, use_masks=None):
        """
        Initialize the dataset.
        
        Args:
            db_name: Database name
            root: Root directory path
            mode: 'train', 'validate', or 'test'
            transform: Transform to apply to images and labels
            desired_height: Target image height
            desired_width: Target image width
            use_masks: Whether to use masks
        """
        self.use_masks = use_masks
        
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
                
        if transform is None:
            self.transform = ResizeTransform(size=(desired_width, desired_height))
        else:
            self.transform = transform
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)

    @staticmethod
    def process_images(image, label):
        """
        Process images and labels for model input.
        
        Args:
            image: PIL image
            label: PIL label image
            
        Returns:
            Normalized and tensor-converted image and label
        """
        # Convert PIL Images to numpy arrays
        image_np = np.array(image, dtype=np.float32)
        label_np = np.array(label, dtype=np.float32)

        # Normalize to [0, 1] range
        if image_np.max() > 1.0:
            image_np /= 255.0
        if label_np.max() > 1.0:
            label_np /= 255.0
            label_np = np.round(label_np)
        else:
            label_np = np.round(label_np)

        # Handle channel dimensions
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.moveaxis(image_np, -1, 0)  # Move channels to first dimension
        if label_np.ndim == 2:
            label_np = np.expand_dims(label_np, 0)  # Add channel dimension
        
        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image_np)
        label_tensor = torch.from_numpy(label_np)
        return image_tensor, label_tensor
    
    def plot_image(self, image, label, title='Image and Label'):
        """
        Plot image and label side by side for visualization.
        
        Args:
            image: Image to plot
            label: Label to plot
            title: Plot title
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.array(image), cmap='gray')
        ax[0].set_title('Image')
        ax[1].imshow(np.array(label), cmap='gray')
        ax[1].set_title('Label')
        plt.suptitle(title)
        plt.show()

    def __getitem__(self, idx):
        """Get a sample from the dataset at the specified index."""
        # Load image and label
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        label = label.convert('L')  # Convert label to grayscale
        
        # Apply transformations
        image, label = self.transform(image, label)
        
        # Process images
        image, label = self.process_images(image, label)

        return image, label