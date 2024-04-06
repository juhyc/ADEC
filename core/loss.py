import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# * Loss functions
###############################################


class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predictions, targets):
        diff = torch.abs(predictions - targets)
        delta = self.threshold * torch.max(diff).item()
        
        # Apply L1 loss, if loss is less then delta
        part1 = diff

        # Appy L2 loss, if loss is greater than delta
        part2 = (diff**2 + delta**2) / (2 * delta)

        loss = torch.where(diff <= delta, part1, part2)
        return loss.mean()


def calculate_entropy(feature_map):
    probs = F.softmax(feature_map, dim = -1)
    
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs, dim = -1)
    return entropy

