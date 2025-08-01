import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', 
                 label_smoothing: float = 0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class (minority class)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / inputs.size(1)
            ce_loss = -torch.sum(targets_one_hot * F.log_softmax(inputs, dim=1), dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities for the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha weights correctly
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss