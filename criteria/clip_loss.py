import torch
import clip
from PIL import Image
import torchvision.transforms as transforms


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class CLIPLoss(torch.nn.Module):

    def __init__(self, direction_loss_type='cosine', distance_loss_type='mae', use_distance=False, src_img_list=None, tar_img_list=None):
        super(CLIPLoss, self).__init__()

    def forward(self, x):
        raise NotImplementedError
