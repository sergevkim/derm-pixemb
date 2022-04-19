import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet3plus import UNet3Plus

BACKBONE = UNet3Plus()


def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += torch.logical_and(prediction[i] > 0, ground_truth[i] > 0).float().sum()
        union += torch.logical_or(prediction[i] > 0, ground_truth[i] > 0).float().sum()
    return float(intersection + 1e-6) / (union + 1e-6)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce_loss = bce_loss + dice_loss

        return dice_bce_loss


#TODO move to Sequentials; einops
class Decoder2Vector(torch.nn.Module):
    def __init__(self, num_out=2):
        super().__init__()
        self.conv = nn.Conv2d(64, 8, 3, 2, 1)
        self.silu = nn.SiLU()
        self.pool = nn.AvgPool2d(16, stride=32)
        self.fc = nn.Linear(128, num_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.silu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class Decoder2Mat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 4, 3, 1, 1)
        self.conv3 = nn.Conv2d(4, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


class Image2VectorWithCE(torch.nn.Module):
    def __init__(self, num_out=2):
        super().__init__()
        self.encoder = BACKBONE
        self.decoder = Decoder2Vector(num_out=num_out)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def compute_loss(self, predictions, gt):
        return F.cross_entropy(predictions, gt).mean()

    def post_processing(self, prediction):
        return prediction.max(1)[1].data

    def metric(self, y_batch, y_pred):
        return (y_batch.cpu() == y_pred.cpu()).float().mean()


class Image2VectorWithMSE(torch.nn.Module):
    def __init__(self, num_out=10):
        super().__init__()
        self.encoder = BACKBONE
        self.decoder = Decoder2Vector(num_out=num_out)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def compute_loss(self, predictions, gt):
        return F.mse_loss(predictions, gt).mean()

    def post_processing(self, prediction):
        return prediction

    def metric(self, y_batch, y_pred):
        return F.mse_loss(y_pred, y_batch).mean()


class Image2Image(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BACKBONE
        self.decoder = Decoder2Mat()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def compute_loss(self, predictions, gt):
        loss=DiceBCELoss()
        return loss(predictions, gt)

    def post_processing(self, prediction):
        return prediction>0.5

    def metric(self, y_batch, y_pred):
        return calc_iou(y_pred, y_batch)

