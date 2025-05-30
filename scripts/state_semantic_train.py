import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from vesicle_process import VesicleSegmentationDataset
from erm_net import UNetResNet18
from sa_unet import SAUNet
from models.state_semantic import U_Net, AttU_Net, NestedUNet
from losses.dice_loss import DiceLoss
from multi_class_confusion import TrainMatrix
from torchmetrics import AUROC


class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, 
                 optimizer: torch.optim.Optimizer, save_every: int, 
                 snapshot_path: str, best_model_path: str) -> None:
        """
        Initialize the trainer with model, data, and training parameters.
        
        Args:
            model: Neural network model to train
            train_data: DataLoader for training data
            optimizer: Optimizer for model parameters
            save_every: Frequency to save model checkpoints
            snapshot_path: Path to save training snapshots
            best_model_path: Path to save best performing model
        """
        self.gpu_id = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0
        self.train_loss_history = []
        self.best_metrics = float('-inf')  # Track best validation metrics
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

        # Initialize loss function for multi-class segmentation
        self.dice_loss_function = DiceLoss(mode='multiclass', classes=(0, 1, 2), from_logits=True)

    def _load_snapshot(self, snapshot_path):
        """Load training state from a snapshot."""
        snapshot = torch.load(snapshot_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_loss_history = snapshot.get("TRAIN_LOSS_HISTORY", [])
        print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def _load_best_model(self, best_model_path):
        """Load best model based on validation metrics."""
        snapshot = torch.load(best_model_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.best_metrics = snapshot["VAL_METRICS"]
        print(f"Best model loaded with metrics: {self.best_metrics}")

    def _run_batch(self, image, label):
        """Process a single batch through the model."""
        final_output = self.model(image)
        loss = self.dice_loss_function(final_output, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def _run_epoch(self, epoch):
        """Run a complete training epoch."""
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
        """Save a snapshot of the current training state."""
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history
        }

        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _save_best_snapshot(self, epoch, val_metrics):
        """Save model if it has the best validation metrics so far."""
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
        """Save validation metrics to a log file."""
        with open(val_metrics_path, 'a') as file:
            file.write(f"Epoch {epoch}: {val_metrics}\n")

    def train(self, max_epochs: int, val_data: DataLoader, val_metrics_path):
        """
        Train the model for specified epochs.
        
        Args:
            max_epochs: Maximum number of epochs to train
            val_data: DataLoader for validation data
            val_metrics_path: Path to save validation metrics log
        """
        for epoch in range(self.epochs_run, max_epochs):
            train_loss = self._run_epoch(epoch)
            val_loss, val_metrics, accuracy = self.validate(val_data)
            print(f"Epoch {epoch} | Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}, Accuracy: {accuracy:.5f}")
            self._save_metrics(epoch, val_metrics, val_metrics_path)
            if val_metrics > self.best_metrics:
                self._save_best_snapshot(epoch, val_metrics)
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def validate(self, val_data: DataLoader):
        """
        Validate model on validation dataset.
        
        Args:
            val_data: Validation data loader
            
        Returns:
            tuple: (average_loss, metrics, accuracy)
        """
        self.model.eval()
        total_loss = 0
        confusion_matrix = TrainMatrix(num_classes=3)
        
        with torch.no_grad():
            for image, label in val_data:
                image = image.to(self.gpu_id)
                label = label.to(self.gpu_id)
                
                final_output = self.model(image)
                loss = self.dice_loss_function(final_output, label)
                
                # Get predictions from model output
                prob_mask = final_output.softmax(dim=1)
                pred_mask = prob_mask.argmax(dim=1)
                
                total_loss += loss.item()
                confusion_matrix.update(label.view(-1), pred_mask.view(-1))
                
        avg_loss = total_loss / len(val_data)
        accuracy, detailed_metrics = confusion_matrix.compute()
        
        return avg_loss, detailed_metrics, accuracy


def load_train_objs(db_name: str, root: str, height: int, width: int, 
                   lr: float, num_classes: int):
    """
    Prepare datasets, model and optimizer for training.
    
    Returns:
        tuple: (train_dataset, val_dataset, model, optimizer)
    """
    train_dataset = VesicleSegmentationDataset(
        db_name, root, mode='train', 
        desired_height=height, desired_width=width
    )
    
    val_dataset = VesicleSegmentationDataset(
        db_name, root, mode='validate',
        desired_height=height, desired_width=width
    )
    
    # Initialize model (using SAUNet for this example)
    state_model = SAUNet()
    
    # Configure optimizer with weight decay
    optimizer = torch.optim.Adam(
        state_model.parameters(), 
        lr=lr, 
        weight_decay=0.0005
    )
    
    return train_dataset, val_dataset, state_model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle=True):
    """Create and configure a DataLoader."""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=True
    )


def main(db_name, root, save_every, total_epochs, batch_size, 
         height, width, lr, snapshot_path, best_model_path, 
         val_metrics_path, num_classes=3):
    """Main training function."""
    
    # Setup datasets, model and optimizer
    train_dataset, val_dataset, model, optimizer = load_train_objs(
        db_name, root, height, width, lr, num_classes
    )
    
    # Prepare data loaders
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size, shuffle=False)
    
    # Initialize and start training
    trainer = Trainer(model, train_data, optimizer, save_every, 
                     snapshot_path, best_model_path)
    trainer.train(total_epochs, val_data, val_metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training setup for semantic segmentation')
    parser.add_argument('--db_name', required=True, type=str, 
                        help='Database name to load the data')
    parser.add_argument('--root', required=True, type=str, 
                        help='Root directory for Training Database')
    parser.add_argument('--total_epochs', type=int, default=200, 
                        help='Total epochs to train')
    parser.add_argument('--save_every', type=int, default=10, 
                        help='Save checkpoint every N epochs')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='Input batch size (default: 10)')
    parser.add_argument('--height', type=int, default=592, 
                        help='Height of the images')
    parser.add_argument('--width', type=int, default=592, 
                        help='Width of the images')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--snapshot_path', type=str, required=True, 
                        help='Path to save model snapshots')
    parser.add_argument('--best_model_path', type=str, required=True, 
                        help='Path to save the best model')
    parser.add_argument('--val_metrics_path', type=str, required=True, 
                        help='Path to store validation metrics')

    args = parser.parse_args()

    # Start training with parsed arguments
    main(args.db_name, args.root, args.save_every, args.total_epochs,
         args.batch_size, args.height, args.width, args.lr,
         args.snapshot_path, args.best_model_path, args.val_metrics_path)
