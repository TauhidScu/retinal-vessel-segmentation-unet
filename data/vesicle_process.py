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
    Returns:
    - data_location (str): The location of the data for the specified database.
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
    Match the images and labels by filename and log mismatches.
    """
    # Get the list of image and label filenames (excluding file extensions)
    image_files = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    label_files = sorted([os.path.splitext(f)[0] for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

    # Log the total number of files
    logger.info(f"Total images found: {len(image_files)}")
    logger.info(f"Total labels found: {len(label_files)}")

    # Ensure both directories have the same number of files and match by name
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


# Example usage
def read_folder(db_name, root, mode, use_masks=None):
    """
    Reads folders for images, preprocessed, labels.

    Parameters:
    - db_name (str): Name of the database.
    - root (str): Root directory location.
    - mode (str): Mode of data ('train', 'validate', 'test').
    - use_masks (bool): Whether to use masks (default is None).

    Returns:
    - images_loc (str): Location of images directory.
    - label_loc (str): Location of labels directory.
    """
    data_location = data_loader(db_name, root)

    if mode == 'train':
        logger.info('Loading train data!')
        if use_masks is not None:
            image_dir = data_location + '/train/images/'
            mask_dir = data_location + '/train/masks/'
            label_dir = data_location + '/train/labels/'
            match_images_and_labels(image_dir, label_dir)  # Check matching images and labels
            return image_dir, mask_dir, label_dir
        else:
            image_dir = data_location + '/train/images/'
            label_dir = data_location + '/train/labels/'
            match_images_and_labels(image_dir, label_dir)  # Check matching images and labels
            return image_dir, label_dir

    elif mode == 'validate':
        logger.info('Loading val data!')
        if use_masks is not None:
            image_dir = data_location + '/validate/images/'
            mask_dir = data_location + '/validate/masks/'
            label_dir = data_location + '/validate/labels/'
            match_images_and_labels(image_dir, label_dir)  # Check matching images and labels
            return image_dir, mask_dir, label_dir
        else:
            image_dir = data_location + '/validate/images/'
            label_dir = data_location + '/validate/labels/'
            match_images_and_labels(image_dir, label_dir)  # Check matching images and labels
            return image_dir, label_dir

    elif mode == 'test':
        logger.info('Loading test data!')
        if use_masks is not None:
            image_dir = data_location + '/test/images/'
            mask_dir = data_location + '/test/masks/'
            label_dir = data_location + '/test/labels/'
            match_images_and_labels(image_dir, label_dir)  # Check matching images and labels
            return image_dir, mask_dir, label_dir
        else:
            image_dir = data_location + '/test/images/'
            label_dir = data_location + '/test/labels/'
            match_images_and_labels(image_dir, label_dir)  # Check matching images and labels
            return image_dir, label_dir
    else:
        logger.error('Data location error!')

class ResizeTransform:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image, label):
        # Resize each image type without changing its channel format
        image = self.resize_image(image)
        label = self.resize_image(label)
        return image, label

    def resize_image(self, img):
        img = self.resize_and_maintain_aspect_ratio(img, self.size)
        img = self.pad_if_smaller(img, self.size, fill=self.fill)
        # Center crop is applied uniformly here; adjust if necessary
        img = self.center_crop(img, self.size)
        return img

    def resize_and_maintain_aspect_ratio(self, img, target_size):
        original_width, original_height = img.size
        target_width, target_height = target_size

        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return img

    def center_crop(self, img, target_size):
        img_width, img_height = img.size
        target_width, target_height = target_size

        crop_width = (img_width - target_width) // 2
        crop_height = (img_height - target_height) // 2

        return img.crop((crop_width, crop_height, crop_width + target_width, crop_height + target_height))

    @staticmethod
    def pad_if_smaller(img, size, fill=0):
        ow, oh = img.size
        padh = size[1] - oh if oh < size[1] else 0
        padw = size[0] - ow if ow < size[0] else 0
        img = ImageOps.expand(img, border=(padw, padh), fill=fill)
        return img

class VesicleSegmentationDataset(Dataset):
    def __init__(self, db_name, root, mode, transform=None, desired_height=1024, desired_width=1536, use_masks=None):
        self.use_masks = use_masks
        self.mode = mode  # Store mode as an attribute
        if self.use_masks is not None:
            images_loc, masks_loc, labels_loc = read_folder(db_name, root, mode, self.use_masks)
            self.images = [os.path.join(images_loc, f) for f in sorted(os.listdir(images_loc)) if os.path.isfile(os.path.join(images_loc, f))]
            self.masks = [os.path.join(masks_loc, f) for f in sorted(os.listdir(masks_loc)) if os.path.isfile(os.path.join(masks_loc, f))]
            self.labels = [os.path.join(labels_loc, f) for f in sorted(os.listdir(labels_loc)) if os.path.isfile(os.path.join(labels_loc, f))]
        else:
            images_loc, labels_loc = read_folder(db_name, root, mode)
            self.images = [os.path.join(images_loc, f) for f in sorted(os.listdir(images_loc)) if os.path.isfile(os.path.join(images_loc, f))]
            self.labels = [os.path.join(labels_loc, f) for f in sorted(os.listdir(labels_loc)) if os.path.isfile(os.path.join(labels_loc, f))]

         # Check that the number of images matches the number of labels
        if len(self.images) != len(self.labels):
            logger.error(f"Number of images: {len(self.images)} does not match number of labels: {len(self.labels)}")
        else:
            logger.info(f"Number of images: {len(self.images)} matches number of labels: {len(self.labels)}")

        
        if transform is None:
            self.transform = ResizeTransform(size=(desired_width, desired_height))
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    @staticmethod
    def process_images(image, label):
        # Convert PIL Images to numpy arrays
        image_np = np.array(image, dtype=np.float32)
        label_np = np.array(label, dtype=np.int64)  # Use int64 for multi-class labels (0, 1, 2, 3)

        #print(f"Max pixel value in original image: {np.max(image)}")
        #print(f"Max pixel value in original label: {np.max(label)}")
        #print('Image shape before processing: ', image_np.shape)       
        #print('Label shape before processing: ', label_np.shape)

        # Normalize the images to [0, 1]
        image_np /= 255.0

        # Handle RGB images (3 channels), convert from HWC to CHW format
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.moveaxis(image_np, -1, 0)  # Moves the last axis to the first position

        # Since the label is already in the correct format, no need for expansion or squeeze
        # The label should remain as [batch_size, height, width]
        #print(f"Max pixel value in processed image: {np.max(image_np)}")
        #print(f"Max pixel value in processed label: {np.max(label_np)}")

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image_np)
        label_tensor = torch.from_numpy(label_np)

        return image_tensor, label_tensor
    
    def plot_image(self, image, label, title='Image and Label'):
        # Convert the image to numpy for plotting
        if image.ndimension() == 3:  # Check if it's a 3-channel image
            image = image.numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        else:  # Grayscale image
            image = image.numpy().squeeze()  # Remove extra channel dimension if grayscale

        label = label.numpy().squeeze()  # Convert label to numpy and remove extra channel if needed

        # Plot image and label side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        ax[0].set_title('Image')
        ax[1].imshow(label, cmap='gray')
        ax[1].set_title('Label')
        plt.suptitle(title)
        plt.show()


    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        label = label.convert('L')  # Convert label to grayscale if it is not already
        
        # Apply the transformation
        image, label = self.transform(image, label)
        
        # Process images (convert to numpy arrays, normalize, convert to tensors)
        image, label = self.process_images(image, label)
        filename = os.path.basename(self.images[idx])
        if self.mode == 'test':
            return image, label, filename

        return image, label

# # Assuming you have defined your dataset and model
# dataset = VesicleSegmentationDataset(db_name='Vesicles', root='D:/a', mode='train')
# image, label = dataset[-6]

# # Plot the first image and label
# dataset.plot_image(image, label)
