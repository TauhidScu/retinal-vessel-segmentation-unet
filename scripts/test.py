import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
import torchvision.transforms as transforms

# Import custom modules
from test_data_process import VesselSegmentationDataset
from state_model import NestedUNet, U_Net, AttU_Net
from erm_net import UNetResNet18
from sa_unet import SAUNet
from confusion_matrix import ConfusionMatrix


def load_model(best_model_path, device):
    """
    Load a pre-trained model from a checkpoint file.
    
    Args:
        best_model_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model on
        
    Returns:
        model: The loaded model in evaluation mode
    """
    # Uncomment the model you want to use
    # model = U_Net()
    # model = AttU_Net()
    # model = NestedUNet()
    model = SAUNet()
    # model = UNetResNet18(n_classes=1)
    
    # Load the model state from the checkpoint
    best_epochs_snapshots = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_epochs_snapshots["MODEL_STATE"])
    model.to(device)
    model.eval()
    return model


def calculate_auc_sklearn(output, label):
    """
    Calculate various evaluation metrics using scikit-learn.
    
    Args:
        output (torch.Tensor): Model predictions
        label (torch.Tensor): Ground truth labels
        
    Returns:
        tuple: Multiple evaluation metrics (auc, sensitivity, specificity, precision, f1, iou, accuracy, mcc)
    """
    # Move tensors to CPU and convert to numpy arrays
    output_np = output.cpu().numpy()
    label_np = label.cpu().numpy()

    # Flatten the arrays
    output_flat = output_np.ravel()
    label_flat = label_np.ravel()

    # Convert labels to binary format (0 or 1)
    label_flat_binary = (label_flat >= 0.5).astype(int)
    
    # Calculate AUC with probability outputs
    auc = roc_auc_score(label_flat_binary, output_flat)

    # Convert predictions to binary for other metrics
    output_flat_binary = (output_flat >= 0.5).astype(int)

    # Calculate confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(label_flat_binary, output_flat_binary).ravel()
    
    # Calculate performance metrics
    sen = tp / (tp + fn) if (tp + fn) else 0  # Sensitivity (Recall)
    spec = tn / (tn + fp) if (tn + fp) else 0  # Specificity
    prec = tp / (tp + fp) if (tp + fp) else 0  # Precision
    f1 = 2 * (prec * sen) / (prec + sen) if (prec + sen) else 0  # F1 Score
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0  # IoU (Jaccard Index)
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0  # Accuracy
    mcc = matthews_corrcoef(label_flat_binary, output_flat_binary)  # Matthews Correlation Coefficient

    return auc, sen, spec, prec, f1, iou, acc, mcc


def test_model(model, test_data_loader, device, save_path):
    """
    Test the model on the test dataset and calculate metrics.
    
    Args:
        model: The model to test
        test_data_loader (DataLoader): DataLoader for test dataset
        device (torch.device): Device to run testing on
        save_path (str): Path to save segmented images
        
    Returns:
        dict: Dictionary containing average values of all metrics
    """
    # Initialize lists to store metrics for each batch
    auroc_values, sen_values, spec_values = [], [], []
    prec_values, f1_values, iou_values = [], [], []
    acc_values, mcc_values = [], []
    image_idx = 0
    
    with torch.no_grad():
        for batch_idx, (image, label, filenames) in enumerate(test_data_loader):
            print(f"\nProcessing Batch: {batch_idx + 1}/{len(test_data_loader)}")  
            
            # Move data to device
            image = image.to(device)
            label = label.to(device)
            
            # Forward pass
            output = model(image)
            
            # Log shapes for debugging
            print(f"Image shape: {image.shape}, Output shape: {output.shape}, Label shape: {label.shape}")
            print(f"Raw Output (first 5 pixels): {output.view(-1)[:5].cpu().numpy()}")

            # Convert output to binary predictions
            preds = (output >= 0.5).to(torch.int64)
            print(f"Binary Output (first 5 pixels): {preds.view(-1)[:5].cpu().numpy()}")
            
            # Calculate metrics
            auroc, sen, spec, prec, fi, iou, acc, sk_mcc = calculate_auc_sklearn(
                output.squeeze(1), label.squeeze(1)
            )
            
            # Store metrics
            auroc_values.append(auroc)
            sen_values.append(sen)
            spec_values.append(spec)
            prec_values.append(prec)
            f1_values.append(fi)
            iou_values.append(iou)
            acc_values.append(acc)
            mcc_values.append(sk_mcc)
            
            # Save segmented images
            save_segmented_images(preds, save_path, filenames)
            image_idx += len(preds)
    
    # Calculate average metrics
    average_metrics = {
        'Accuracy': sum(acc_values) / len(acc_values),
        'AUROC': sum(auroc_values) / len(auroc_values),
        'Sensitivity': sum(sen_values) / len(sen_values),
        'Specificity': sum(spec_values) / len(spec_values),
        'Precision': sum(prec_values) / len(prec_values),
        'f1_score': sum(f1_values) / len(f1_values),
        'IoU': sum(iou_values) / len(iou_values),
        'MCC': sum(mcc_values) / len(mcc_values)
    }
    
    return average_metrics


def save_segmented_images(predictions, save_path, filenames):
    """
    Save segmented images with original filenames.

    Args:
        predictions (torch.Tensor): Batch of predicted masks (B, H, W)
        save_path (str): Directory path to save the segmented images
        filenames (list): List of filenames corresponding to the original images
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, pred in enumerate(predictions):
        # Convert prediction to a suitable format for saving (2D, uint8)
        pred_enhanced = pred.squeeze().byte().cpu().numpy() * 255

        # Log information about the image being saved
        print(f"Saving: {filenames[i]}, Shape: {pred_enhanced.shape}, Type: {pred_enhanced.dtype}")

        # Create filename with "_segmented" suffix
        segmented_filename = f"{os.path.splitext(filenames[i])[0]}_segmented.png"
        segmented_filepath = os.path.join(save_path, segmented_filename)

        # Save the segmented image
        Image.fromarray(pred_enhanced).save(segmented_filepath)


def main(best_model_path, db_name, root, save_path, metrics_save_path, height, width, test_data_size):
    """
    Main function to execute the testing pipeline.
    
    Args:
        best_model_path (str): Path to the saved model checkpoint
        db_name (str): Database name for test data
        root (str): Root directory path for data
        save_path (str): Path to save segmented images
        metrics_save_path (str): Path to save metrics text file
        height (int): Height of the test images
        width (int): Width of the test images
        test_data_size (int): Batch size for testing
    """
    # Set device (GPU or CPU)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model(best_model_path, device)

    # Create the test dataset and dataloader
    test_dataset = VesselSegmentationDataset(db_name, root, mode='test', 
                                            desired_height=height, desired_width=width)
    test_data_loader = DataLoader(test_dataset, batch_size=test_data_size, shuffle=False)

    # Test the model and get metrics
    metrics = test_model(model, test_data_loader, device, save_path)
    print("Test Metrics:", metrics)
    
    # Ensure directory exists for metrics_save_path
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    
    # Save metrics to a text file
    with open(metrics_save_path, 'w') as file:
        for key, value in metrics.items():
            file.write(f'{key}: {value}\n')


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--best_model_path', type=str, required=True, help='Path to the saved model snapshot')
    parser.add_argument('--db_name', type=str, required=True, help='Database name for test data')
    parser.add_argument('--root', type=str, required=True, help='Path to the root directory to map database location')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save segmented images')
    parser.add_argument('--metrics_save_path', type=str, required=True, help='Path to save metrics text file')
    parser.add_argument('--height', type=int, default=592, help='Height of the test images for specified database')
    parser.add_argument('--width', type=int, default=592, help='Width of the test images for specified database')
    parser.add_argument('--test_data_size', type=int, default=20, help='Batch size for testing')

    args = parser.parse_args()
    
    # Execute the main function
    main(args.best_model_path, args.db_name, args.root, args.save_path, 
         args.metrics_save_path, args.height, args.width, args.test_data_size)
