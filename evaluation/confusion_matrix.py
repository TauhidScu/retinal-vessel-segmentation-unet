import torch


class ConfusionMatrix(object):
    """
    Class for computing confusion matrix and related evaluation metrics.
    """
    def __init__(self, num_classes):
        """
        Initialize confusion matrix tracker.
        
        Args:
            num_classes: Total number of classes
        """
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        """
        Update confusion matrix with batch of actual and predicted labels.
        
        Args:
            a: Ground truth labels
            b: Predicted labels
        """
        n = self.num_classes
        if self.mat is None:
            # Initialize confusion matrix on the same device as input
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        
        with torch.no_grad():
            # Calculate indices for valid entries
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            # Update counts
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        """
        Compute evaluation metrics from the confusion matrix.
        
        Returns:
            Dictionary containing accuracy and averaged metrics
        """
        h = self.mat.float()
        acc_global = (torch.diag(h).sum() / h.sum()).item()

        # Initialize sums for metrics
        sum_sensitivity = 0
        sum_specificity = 0
        sum_precision = 0
        sum_f1_score = 0
        sum_iou = 0

        # Calculate metrics for each class
        for class_idx in range(self.num_classes):
            TP = h[class_idx, class_idx]
            FP = h[:, class_idx].sum() - TP
            FN = h[class_idx, :].sum() - TP
            TN = h.sum() - (FP + FN + TP)

            # Calculate metrics for the current class
            specificity = TN / (TN + FP) if TN + FP > 0 else 0
            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0
            iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0

            # Aggregate metrics for averaging
            sum_sensitivity += sensitivity
            sum_specificity += specificity
            sum_precision += precision
            sum_f1_score += f1_score
            sum_iou += iou

        # Average metrics over all classes
        avg_sensitivity = sum_sensitivity / self.num_classes
        avg_specificity = sum_specificity / self.num_classes
        avg_precision = sum_precision / self.num_classes
        avg_f1_score = sum_f1_score / self.num_classes
        avg_iou = sum_iou / self.num_classes

        # Return averaged metrics
        return {
            'accuracy': acc_global,
            'average_sensitivity': avg_sensitivity,
            'average_specificity': avg_specificity,
            'average_precision': avg_precision,
            'average_f1_score': avg_f1_score,
            'average_iou': avg_iou
        }


class TrainMatrix(object):
    """
    Simplified confusion matrix for tracking training metrics.
    """
    def __init__(self, num_classes):
        """
        Initialize training metrics tracker.
        
        Args:
            num_classes: Total number of classes
        """
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        """
        Update matrix with batch of actual and predicted labels.
        
        Args:
            a: Ground truth labels
            b: Predicted labels
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    
    def compute(self):
        """
        Compute accuracy and F1 score from the matrix.
        
        Returns:
            tuple: (accuracy, average F1 score)
        """
        if self.mat is None:
            return 0.0, 0.0
            
        h = self.mat.float()
        acc_global = (torch.diag(h).sum() / h.sum()).item()
        sum_f1_score = 0

        for class_idx in range(self.num_classes):
            TP = h[class_idx, class_idx]
            FP = h[:, class_idx].sum() - TP
            FN = h[class_idx, :].sum() - TP

            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0
            sum_f1_score += f1_score

        avg_f1_score = (sum_f1_score / self.num_classes).item()
        return acc_global, avg_f1_score  # Return as floats

    def reset(self):
        """
        Reset the matrix to None.
        """
        self.mat = None