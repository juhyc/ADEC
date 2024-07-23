import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

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


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input, target):
        input = self.transform(input)
        target = self.transform(target)
        input_features = self.layers(input)
        target_features = self.layers(target)
        loss = nn.functional.l1_loss(input_features, target_features)
        return loss


def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx

def gradient_y(img):
    gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gy

def gradient_loss(pred, target):
    pred_gradient_x = gradient_x(pred)
    target_gradient_x = gradient_x(target)
    pred_gradient_y = gradient_y(pred)
    target_gradient_y = gradient_y(target)
    loss_x = torch.mean(torch.abs(pred_gradient_x - target_gradient_x))
    loss_y = torch.mean(torch.abs(pred_gradient_y - target_gradient_y))
    return loss_x + loss_y

def smoothness_loss(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
