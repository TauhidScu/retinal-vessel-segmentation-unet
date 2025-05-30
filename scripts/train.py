import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from data.preprocess import VesselSegmentationDataset
from models.erm_net import UNetResNet18
from models.sa_unet import SAUNet
from models.state_model import NestedUNet, U_Net, AttU_Net
from losses.loss import *
from evaluation.confusion_matrix import TrainMatrix
from torchmetrics import AUROC


class Trainer:
    """Trainer class for managing model training, validation, and checkpointing."""

    def __init__(self, model: torch.nn.Module, train_data: DataLoader, 
                optimizer: torch.optim.Optimizer, save_every: int, 
                snapshot_path: str, best_model_path: str) -> None:
        """Initialize the trainer with model, data, and configuration."""
        self.gpu_id = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0
        self.train_loss_history = []
        self.best_metrics = float('-inf') 
        self.best_model_path = best_model_path 
        
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        # Load existing checkpoints if available
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        if os.path.exists(self.best_model_path):
            print("Loading best model")
            self._load_best_model(self.best_model_path)
        
        # Initialize loss functions
        self.bce_loss_function = BCELoss()
        self.dice_loss_function = DiceLoss()
        self.combined_loss = CombinedLoss(bce_loss_weight=0.25, dice_loss_weight=0.75)

    def _load_snapshot(self, snapshot_path):
        """Load training state from a snapshot file."""
        snapshot = torch.load(snapshot_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_loss_history = snapshot.get("TRAIN_LOSS_HISTORY", [])
        print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def _load_best_model(self, best_model_path):
        """Load the best model based on validation metrics."""
        snapshot = torch.load(best_model_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.best_metrics = snapshot["VAL_METRICS"]
        print(f"Best model loaded with metrics: {self.best_metrics}")

    def _run_batch(self, image, label):
        """Process a single batch through the model."""
        final_output = self.model(image)
        loss = self.combined_loss(final_output, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def _run_epoch(self, epoch):
        """Run a complete epoch of training."""
        self.model.train()
        running_loss = 0
        for image, label in self.train_data:
            image, label = image.to(self.gpu_id), label.to(self.gpu_id)
            batch_loss = self._run_batch(image, label)
            running_loss += batch_loss
        avg_loss = running_loss / len(self.train_data)
        self.train_loss_history.append(avg_loss)
        return avg_loss

    def _save_snapshot(self, epoch):
        """Save current training state as a snapshot."""
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _save_best_snapshot(self, epoch, val_metrics):
        """Save model as the best if it has improved validation metrics."""
        if val_metrics > self.best_metrics:
            self.best_metrics = val_metrics
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "EPOCHS_RUN": epoch,
                "VAL_METRICS": val_metrics
            }
            torch.save(snapshot, self.best_model_path)
            print(f"Epoch {epoch} | Best model saved with metrics {val_metrics:.5f}")

    def _save_metrics(self, epoch, val_metrics, val_metrics_path):
        """Save validation metrics to a file."""
        with open(val_metrics_path, 'a') as file:
            file.write(f"Epoch {epoch}: {val_metrics}\n")

    def train(self, max_epochs: int, val_data: DataLoader, val_metrics_path):
        """Main training loop that runs for specified epochs."""
        for epoch in range(self.epochs_run, max_epochs):
            train_loss = self._run_epoch(epoch)
            val_loss, val_metrics, accuracy = self.validate(val_data)
            
            # Print epoch results
            print(f"Epoch {epoch} | Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}, Accuracy: {accuracy:.5f}")
            
            # Save metrics and model checkpoints
            self._save_metrics(epoch, val_metrics, val_metrics_path)
            if val_metrics > self.best_metrics:
                self._save_best_snapshot(epoch, val_metrics)
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def validate(self, val_data: DataLoader, threshold_labels=False):
        """Validate model on validation dataset."""
        self.model.eval()
        total_loss = 0.0
        confusion_matrix = TrainMatrix(num_classes=2)
        
        with torch.no_grad():
            for image, label in val_data:
                image = image.to(self.gpu_id)
                label = label.to(self.gpu_id)
                
                # Forward pass
                final_output = self.model(image)
                preds = (final_output >= 0.5).to(torch.int64)
                
                # Calculate loss
                loss = self.combined_loss(final_output, label)
                total_loss += loss.item()
                
                # Process labels based on threshold setting
                if threshold_labels:
                    binary_label = (label >= 0.5).to(torch.int64)
                else:
                    binary_label = label.to(torch.int64)

                # Update confusion matrix with predictions
                confusion_matrix.update(binary_label.view(-1), preds.view(-1))
                
        # Calculate average metrics
        avg_loss = total_loss / len(val_data)
        train_accuracy, avg_f1_score = confusion_matrix.compute()
        
        # Reset confusion matrix for next validation
        confusion_matrix.reset()

        return avg_loss, avg_f1_score, train_accuracy


def load_train_objs(db_name: str, root: str, height: int, width: int, lr: float):
    """Load datasets, model, and optimizer for training."""
    # Create datasets
    train_dataset = VesselSegmentationDataset(
        db_name, root, mode='train', desired_height=height, desired_width=width
    )
    val_dataset = VesselSegmentationDataset(
        db_name, root, mode='validate', desired_height=height, desired_width=width
    )
    
    # Initialize model - uncomment the model you want to use
    # state_model = U_Net()
    # state_model = AttU_Net()
    # state_model = NestedUNet()
    state_model = SAUNet()
    # state_model = UNetResNet18(n_classes=1, block_size=7, keep_prob=0.9, sync_channels=False)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(state_model.parameters(), lr=lr, weight_decay=0.0005)
    
    return train_dataset, val_dataset, state_model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle=True):
    """Create data loader from dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def main(db_name, root, save_every, total_epochs, batch_size, height, width, 
        lr, snapshot_path, best_model_path, val_metrics_path):
    """Main function to set up and start the training process."""
    # Load training objects
    train_dataset, val_dataset, model, optimizer = load_train_objs(
        db_name, root, height, width, lr
    )
    
    # Prepare data loaders
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size, shuffle=False)
    
    # Initialize trainer and start training
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, best_model_path)
    trainer.train(total_epochs, val_data, val_metrics_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Training setup')
    parser.add_argument('--db_name', required=True, type=str, help='Database name to load the data')
    parser.add_argument('--root', required=True, type=str, help='Root directory for Training Database')
    parser.add_argument('--total_epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Count to save snapshot')
    parser.add_argument('--batch_size', type=int, default=10, help='Input batch size default: 10')
    parser.add_argument('--height', type=int, default=592, help='Height of the images')
    parser.add_argument('--width', type=int, default=592, help='Width of the images')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--snapshot_path', type=str, required=True, help='Path to save model snapshots')
    parser.add_argument('--best_model_path', type=str, required=True, help='Path to save the best model')
    parser.add_argument('--val_metrics_path', type=str, required=True, help='File(txt)Path to save validation metrics')

    # Parse arguments and start training
    args = parser.parse_args()
    main(args.db_name, args.root, args.save_every, args.total_epochs, args.batch_size, 
         args.height, args.width, args.lr, args.snapshot_path, args.best_model_path, 
         args.val_metrics_path)
