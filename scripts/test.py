import torch
from torch.utils.data import DataLoader
from test_data_process import VesselSegmentationDataset
#from state_model import AttU_Net
from state_model import NestedUNet, U_Net, AttU_Net
from erm_net import UNetResNet18
from sa_unet import SAUNet
from confusion_matrix import ConfusionMatrix
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
import torchvision.transforms as transforms


def load_model(best_model_path, device):
    #model = U_Net()
    #model = AttU_Net() #NestedUNet() #AttU_Net() #NestedUNet() #
    model = SAUNet()
    #model = UNetResNet18(n_classes=1)
    best_epochs_snapshots = torch.load(best_model_path, map_location=device)
    # Load the model state
    model.load_state_dict(best_epochs_snapshots["MODEL_STATE"])
    model.to(device)
    model.eval()
    return model

def calculate_auc_sklearn(output, label):
    # Ensure the output and label tensors are on the CPU and convert to numpy arrays
    print('output shape: ', output.shape)
    print('label shape: ', label.shape)

    output_np = output.cpu().numpy()
    label_np = label.cpu().numpy()

    # Flatten the arrays
    output_flat = output_np.ravel()
    label_flat = label_np.ravel()

    # Ensure the labels are binary (0 or 1)
    label_flat_binary = (label_flat >= 0.5).astype(int)

    # Calculate AUC with probabilities as output
    auc = roc_auc_score(label_flat_binary, output_flat)

    # Convert predictions to binary for other metrics
    output_flat_binary = (output_flat >= 0.5).astype(int)

    # Calculate other metrics
    mcc = matthews_corrcoef(label_flat_binary, output_flat_binary)
    tn, fp, fn, tp = confusion_matrix(label_flat_binary, output_flat_binary).ravel()
    sen = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * (prec * sen) / (prec + sen) if (prec + sen) else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0

    return auc, sen, spec, prec, f1, iou, acc, mcc



def test_model(model, test_data_loader, device, save_path):
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
            # print('output shape: ', output.shape)
            # print('label shape: ', label.shape)

            # # Convert output to binary predictions
            preds = (output >= 0.5).to(torch.int64)
            print(f"Raw Output (first 5 pixels): {preds.view(-1)[:5].cpu().numpy()}")
            #binary_label = (label >= 0.5).to(torch.int64)
            auroc, sen, spec, prec, fi, iou, acc, sk_mcc = calculate_auc_sklearn(output.squeeze(1), label.squeeze(1))
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
                     
    average_auroc = sum(auroc_values) / len(auroc_values)
    average_sen = sum(sen_values) / len(sen_values)
    average_spec = sum(spec_values) / len(spec_values)
    average_prec = sum(prec_values) / len(prec_values)
    average_f1 = sum(f1_values) / len(f1_values)
    average_iou = sum(iou_values) / len(iou_values)
    average_acc = sum(acc_values) / len(acc_values)
    average_mcc = sum(mcc_values) / len(mcc_values)
    return {'Accuracy': average_acc, 'AUROC': average_auroc, 'Sensitivity': average_sen, 'Specificity': average_spec, 'Precision': average_prec, 'f1_score': average_f1, 'IoU': average_iou, 'MCC': average_mcc}

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

    for i, pred in enumerate(predictions):
        # Ensure the prediction is squeezed to 2D (H, W) and in uint8 format
        pred_enhanced = pred.squeeze().byte().cpu().numpy() * 255

        # Check the shape and type before saving
        print(f"Saving: {filenames[i]}, Shape: {pred_enhanced.shape}, Type: {pred_enhanced.dtype}")

        # Create filename with "_segmented" suffix
        segmented_filename = f"{os.path.splitext(filenames[i])}_segmented.png"
        segmented_filepath = os.path.join(save_path, segmented_filename)

        # Save the segmented image
        Image.fromarray(pred_enhanced).save(segmented_filepath)

def main(best_model_path:str, db_name:str, root:str, save_path:str, metrics_save_path:str, height:int, width:int, test_data_size: int):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = load_model(best_model_path, device)

    test_dataset = VesselSegmentationDataset(db_name, root, mode='test', desired_height=height, desired_width=width)
    test_data_loader = DataLoader(test_dataset, batch_size=test_data_size, shuffle=False)

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
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--best_model_path', type=str, required=True, help='path to the saved model snapshot')
    parser.add_argument('--db_name', type=str, required=True, help='db_name for test data')
    parser.add_argument('--root', type=str, required=True, help='path to the root directory to map database location')
    parser.add_argument('--save_path', type=str, required=True, help='path to save segmented images')
    parser.add_argument('--metrics_save_path', type=str, required=True, help='path to save metrics text file')
    parser.add_argument('--height', type=int, default=592, help='Height of the test images for Specified Database')
    parser.add_argument('--width', type=int, default=592, help='Width of the test images for Specified Database')
    parser.add_argument('--test_data_size', type=int, default=20, help='total number test images')

    args = parser.parse_args()
    main(args.best_model_path, args.db_name, args.root, args.save_path, args.metrics_save_path, args.height, args.width, args.test_data_size)
