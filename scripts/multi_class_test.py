import torch
import numpy as np
from torch.utils.data import DataLoader
from vesicle_process import VesicleSegmentationDataset
#from state_model import AttU_Net
from models.state_semantic import NestedUNet, U_Net, AttU_Net
from erm_net import UNetResNet18
from sa_unet import SAUNet
from multi_class_confusion import ConfusionMatrix
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
import torchvision.transforms as transforms


def load_model(best_model_path, device):
    #model = U_Net()
    #model = AttU_Net() #NestedUNet() #
    #model = NestedUNet() #SAUNet()
    model = SAUNet() #UNetResNet18()
    best_epochs_snapshots = torch.load(best_model_path, map_location=device)
    # Load the model state
    model.load_state_dict(best_epochs_snapshots["MODEL_STATE"])
    model.to(device)
    model.eval()
    return model
def test_model(model, test_data_loader, device, save_path, num_classes=3):
    # Initialize confusion matrix for multiclass
    confusion_matrix = ConfusionMatrix(num_classes)
    auroc_values = []
    sen_values = []
    spec_values = []
    prec_values = []
    f1_values = []
    iou_values = []
    acc_values = []
    mcc_values = []

    image_idx = 0
    with torch.no_grad():
        for batch_idx, (image, label, filenames) in enumerate(test_data_loader):
            print(f"\nProcessing Batch: {batch_idx + 1}/{len(test_data_loader)}")  
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            # Ensure correct tensor dimensions
            print(f"Image shape: {image.shape}, Output shape: {output.shape}, Label shape: {label.shape}")
            # Print the raw output for inspection
            print(f"Raw Output (first 5 pixels): {output.view(-1)[:5].cpu().numpy()}")
            preds = output.softmax(dim=1)            
            # Convert output to multiclass predictions
            preds = preds.argmax(dim=1)  # Get the class with the highest score for each pixel
            print(f"Predictions (first 5 pixels): {preds.view(-1)[:5].cpu().numpy()}")

            # Update confusion matrix with ground truth and predictions
            confusion_matrix.update(label, preds)

            # Save segmented images
            save_segmented_images(preds, save_path, filenames)
            image_idx += len(preds)
        
    # Compute multiclass metrics from the confusion matrix
    confusion_metrics = confusion_matrix.compute()
    return confusion_metrics


def save_segmented_images(predictions, save_path, filenames):
    """
    Save segmented images with original filenames.

    Parameters:
    - predictions (torch.Tensor): Batch of predicted masks (B, H, W).
    - save_path (str): Directory path to save the segmented images.
    - filenames (list): List of filenames corresponding to the original images.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define a color map for the 4 classes (each class is represented by an RGB color)
    color_map = {
        0: (0, 0, 0),         # Class 0 - Black
        1: (255, 0, 0),       # Class 1 - Red
        2: (0, 255, 0),       # Class 2 - Green
    }
    class_priority = [0, 2, 1]

    for i, pred in enumerate(predictions):
        # Ensure the prediction is squeezed to 2D (H, W)
        pred = pred.squeeze().cpu().numpy()

        # Create an empty image with RGB channels (H, W, 3)
        h, w = pred.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Map each class to its corresponding RGB color
        for class_id in class_priority:  
            if class_id in color_map:
                color_img[pred == class_id] = color_map[class_id]

        # Check the shape and type before saving
        print(f"Saving: {filenames[i]}, Shape: {color_img.shape}, Type: {color_img.dtype}")

        # Create filename with "_segmented" suffix
        segmented_filename = f"{os.path.splitext(filenames[i])[0]}_segmented.png"
        segmented_filepath = os.path.join(save_path, segmented_filename)

        # Save the color-mapped segmented image
        Image.fromarray(color_img).save(segmented_filepath)


def main(best_model_path:str, db_name:str, root:str, save_path:str, metrics_save_path:str, height:int, width:int, test_data_size: int, num_classes=3):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = load_model(best_model_path, device)

    test_dataset = VesicleSegmentationDataset(db_name, root, mode='test', desired_height=height, desired_width=width)
    test_data_loader = DataLoader(test_dataset, batch_size=test_data_size, shuffle=False)

    # Pass num_classes to the test_model function
    metrics = test_model(model, test_data_loader, device, save_path, num_classes)
    # Print metrics in key-value pair format in the main function
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")  # You can adjust the precision (e.g., .4f) as needed

    # Ensure directory exists for metrics_save_path
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    
    # Save metrics to a text file
    with open(metrics_save_path, 'w') as file:
        for key, value in metrics.items():
            file.write(f'{key}: {value}\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--best_model_path', type=str, required=True, help='path to the saved model snapshot')
    parser.add_argument('--db_name', type=str, required=True, help='db_name for test data')
    parser.add_argument('--root', type=str, required=True, help='path to the root directory to map database location')
    parser.add_argument('--save_path', type=str, required=True, help='path to save segmented images')
    parser.add_argument('--metrics_save_path', type=str, required=True, help='path to save metrics text file')
    parser.add_argument('--height', type=int, default=592, help='Height of the test images for Specified Database')
    parser.add_argument('--width', type=int, default=592, help='Width of the test images for Specified Database')
    parser.add_argument('--test_data_size', type=int, default=20, help='total number test images')
    parser.add_argument('--num_classes', type=int, default=4, help='Number classes to segment: 10')
    args = parser.parse_args()
    main(args.best_model_path, args.db_name, args.root, args.save_path, args.metrics_save_path, args.height, args.width, args.test_data_size)


