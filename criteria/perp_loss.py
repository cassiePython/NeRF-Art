import torch
import torch.nn as nn
from torchvision.models import vgg16

__all__ = ['VGGPerceptualLoss']


# VGG loss, Cite from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()

    def forward(self, x):
        raise NotImplementedError
