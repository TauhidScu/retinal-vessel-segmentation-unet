import torch


class ConfusionMatrix:
    """
    A class to compute and store confusion matrix for multi-class classification or segmentation.
    Calculates various metrics including accuracy, sensitivity, specificity, precision, F1-score, and IoU.
    """
    
    def __init__(self, num_classes):
        """
        Initialize the confusion matrix.
        
        Args:
            num_classes: Total number of classes in the dataset
        """
        self.num_classes = num_classes
        self.mat = None
    
    def update(self, a, b):
        """
        Update confusion matrix with batch of predictions.
        
        Args:
            a: Ground truth labels
            b: Predicted labels
        """
        n = self.num_classes
        
        # Initialize matrix if this is the first update
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
            
        with torch.no_grad():
            # Filter valid entries (classes within range)
            k = (a >= 0) & (a < n)
            
            # Calculate linear indices in the flattened matrix
            inds = n * a[k].to(torch.int64) + b[k]
            
            # Update confusion matrix counts
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    
    def compute(self):
        """
        Compute metrics based on the accumulated confusion matrix.
        
        Returns:
            Dictionary containing accuracy, sensitivity, specificity, precision, F1-score and IoU
            averaged over all classes
        """
        # Convert to float for division operations
        h = self.mat.float()
        
        # Calculate global accuracy
        acc_global = (torch.diag(h).sum() / h.sum()).item()
        
        # Initialize sums for averaging metrics
        sum_sensitivity = 0
        sum_specificity = 0
        sum_precision = 0
        sum_f1_score = 0
        sum_iou = 0
        
        # Calculate metrics for each class
        for class_idx in range(self.num_classes):
            # Extract confusion matrix components for current class
            TP = h[class_idx, class_idx]                  # True Positives
            FP = h[:, class_idx].sum() - TP               # False Positives
            FN = h[class_idx, :].sum() - TP               # False Negatives
            TN = h.sum() - (FP + FN + TP)                 # True Negatives
            
            # Calculate per-class metrics with zero-division handling
            specificity = TN / (TN + FP) if TN + FP > 0 else 0
            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0
            iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
            
            # Accumulate metrics
            sum_sensitivity += sensitivity
            sum_specificity += specificity
            sum_precision += precision
            sum_f1_score += f1_score
            sum_iou += iou
        
        # Calculate class-averaged metrics
        avg_sensitivity = sum_sensitivity / self.num_classes
        avg_specificity = sum_specificity / self.num_classes
        avg_precision = sum_precision / self.num_classes
        avg_f1_score = sum_f1_score / self.num_classes
        avg_iou = sum_iou / self.num_classes
        
        # Return all metrics
        return {
            'accuracy': acc_global,
            'average_sensitivity': avg_sensitivity,
            'average_specificity': avg_specificity,
            'average_precision': avg_precision,
            'average_f1_score': avg_f1_score,
            'average_iou': avg_iou
        }


class TrainMatrix:
    """
    A simplified confusion matrix for training that only computes accuracy and F1-score.
    Optimized for training loop efficiency.
    """
    
    def __init__(self, num_classes):
        """
        Initialize the training matrix.
        
        Args:
            num_classes: Total number of classes in the dataset
        """
        self.num_classes = num_classes
        self.mat = None
    
    def update(self, a, b):
        """
        Update training matrix with batch of predictions.
        
        Args:
            a: Ground truth labels
            b: Predicted labels
        """
        n = self.num_classes
        
        # Initialize matrix if this is the first update
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
            
        with torch.no_grad():
            # Filter valid entries
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    
    def compute(self):
        """
        Compute accuracy and F1-score based on the accumulated data.
        
        Returns:
            Tuple of (accuracy, F1-score)
        """
        h = self.mat.float()
        
        # Calculate global accuracy
        acc_global = (torch.diag(h).sum() / h.sum()).item()
        
        # Calculate class-wise F1-scores
        sum_f1_score = 0
        for class_idx in range(self.num_classes):
            TP = h[class_idx, class_idx]
            FP = h[:, class_idx].sum() - TP
            FN = h[class_idx, :].sum() - TP
            
            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0
            sum_f1_score += f1_score
        
        # Average F1-score across classes
        avg_f1_score = (sum_f1_score / self.num_classes).item()
        
        return acc_global, avg_f1_score


# Example usage (commented out)
"""
# Example with multi-class segmentation (4 classes: 0, 1, 2, 3)
num_classes = 4
confusion_matrix = ConfusionMatrix(num_classes)
train_matrix = TrainMatrix(num_classes)

# Define dummy data
actual_labels = torch.tensor([0, 1, 2, 1, 0, 1, 2, 3, 0, 1])
predicted_labels = torch.tensor([0, 0, 2, 1, 0, 1, 3, 2, 0, 2])

# Update matrices
confusion_matrix.update(actual_labels, predicted_labels)
train_matrix.update(actual_labels, predicted_labels)

# Compute metrics
confusion_metrics = confusion_matrix.compute()
train_accuracy, train_f1_score = train_matrix.compute()

# Print results
print("Confusion Matrix Metrics:")
for metric, value in confusion_metrics.items():
    print(f"{metric}: {value:.4f}")

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Train F1 Score: {train_f1_score:.4f}")
"""
