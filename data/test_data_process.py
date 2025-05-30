import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def data_loader(db_name: str, root: str) -> str:
    """
    Returns:
    - data_location (str): The location of the data for the specified database.
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

#function to read image and label folder
def read_folder(db_name, root, mode, use_masks=None):
    """
    Reads folders for images, preprocessed, labels.

    Parameters:
    - db_name (str): Name of the database.

    Returns:
    - images_loc (str): Location of training images.
    - label_loc (str): Location of training labels.
   
    """
    data_location = data_loader(db_name, root)
    if mode == 'train':    
        print('Lodaing train data!')
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
        print('Lodaing val data!')     
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
        print('Lodaing test data!')
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


#Transform classes for resizing and cropping
class ResizeTransform:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image, label):
        if image.size == self.size and label.size == self.size:
            return image, label
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

        img = img.resize((new_width, new_height), Image.ANTIALIAS)

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

class VesselSegmentationDataset(Dataset):
    def __init__(self, db_name, root, mode, transform=None, desired_height=1024, desired_width=1536, use_masks = None):
        self.use_masks = use_masks
        if self.use_masks is not None:
            images_loc, masks_loc, labels_loc = read_folder(db_name, root, mode, self.use_masks)
            self.images = [os.path.join(images_loc, f) for f in sorted(os.listdir(images_loc)) if os.path.isfile(os.path.join(images_loc, f))]
            self.masks = [os.path.join(masks_loc, f) for f in sorted(os.listdir(masks_loc)) if os.path.isfile(os.path.join(masks_loc, f))]
            self.labels = [os.path.join(labels_loc, f) for f in sorted(os.listdir(labels_loc)) if os.path.isfile(os.path.join(labels_loc, f))]
        else:
            images_loc, labels_loc = read_folder(db_name, root, mode)
            self.images = [os.path.join(images_loc, f) for f in sorted(os.listdir(images_loc)) if os.path.isfile(os.path.join(images_loc, f))]
            self.labels = [os.path.join(labels_loc, f) for f in sorted(os.listdir(labels_loc)) if os.path.isfile(os.path.join(labels_loc, f))]
        if transform is None:
            self.transform = ResizeTransform(size=(desired_width, desired_height))
        else:
            self.transform = transform
        #self.transform = transform or ResizeTransform(size=(desired_width, desired_height))
    def __len__(self):
        return len(self.images)

    @staticmethod
    def process_images(image, label):
        # Convert PIL Images to numpy arrays
        image_np = np.array(image, dtype=np.float32)
        label_np = np.array(label, dtype=np.float32)
        if image_np.max() > 1.0:
            image_np /= 255.0
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
    
    def plot_image(self, image, label, title = 'Image and Label'):
        #Plot image and label side by side
        fig, ax = plt.subplots(1, 2, figsize= (10, 5))
        ax[0].imshow(np.array(image), cmap = 'gray')
        ax[0].set_title('Image')
        ax[1].imshow(np.array(label), cmap = 'gray')
        ax[1].set_title('Label')
        plt.suptitle(title)
        plt.show()


    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx]).convert('L')
        
        #Plot the original image
        #self.plot_image(image, label, title='Original image')
        # Apply the transformation
        image, label = self.transform(image, label)
        #Plot the image and label after transformation
        #self.plot_image(image, label, title='Transformed image')
        # Process images (convert to numpy arrays, normalize, convert to tensors)
        image, label = self.process_images(image, label)
        filename = os.path.basename(self.images[idx])
        # Plot the processed image and label (converted back to numpy for plotting)
        #self.plot_image(image.numpy().transpose(1, 2, 0), label.numpy().squeeze(0), title="Processed Image and Label")


        return image, label, filename