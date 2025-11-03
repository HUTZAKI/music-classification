import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard-to-classify examples

    Paper: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
                   Higher gamma = more focus on hard examples
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions [batch_size, num_classes] (logits)
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights
    Combines focal loss with class-specific weights
    """

    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        """
        Args:
            class_weights: Tensor of shape [num_classes] with class weights
            gamma: Focusing parameter
            reduction: 'mean' or 'sum'
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions [batch_size, num_classes] (logits)
            targets: Ground truth labels [batch_size]

        Returns:
            Weighted focal loss value
        """
        # Get class weights for each sample
        if self.class_weights is not None:
            weights = self.class_weights[targets]
        else:
            weights = torch.ones_like(targets, dtype=torch.float)

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Compute focal loss with weights
        focal_loss = weights * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calculate_class_weights(dataset, num_classes, device='cpu'):
    """
    Calculate class weights based on inverse frequency

    Args:
        dataset: Dataset object with labels
        num_classes: Number of classes
        device: Device to put weights on

    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label] += 1

    # Calculate weights (inverse frequency)
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)

    # Handle classes with 0 samples
    class_weights[class_counts == 0] = 0

    print(f"\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {int(count)} samples, weight: {class_weights[i]:.4f}")

    return class_weights.to(device)
