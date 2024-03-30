import torch
import torch.nn as nn
import torch.nn.functional as F

class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predictions, targets):
        diff = torch.abs(predictions - targets)
        delta = self.threshold * torch.max(diff).item()

        # 델타보다 작은 오류에 대해서는 L1 손실 사용
        part1 = diff

        # 델타보다 큰 오류에 대해서는 델타에 의해 수정된 L2 손실 사용
        part2 = (diff**2 + delta**2) / (2 * delta)

        # 두 부분을 결합
        loss = torch.where(diff <= delta, part1, part2)
        return loss.mean()


def calculate_entropy(feature_map):
    probs = F.softmax(feature_map, dim = -1)
    
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs, dim = -1)
    return entropy

