import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0, distance_type='euclidean'):
        super(ContrastiveLoss, self).__init__()

    def forward(self, x):
        raise NotImplementedError