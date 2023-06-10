import os
import torch
from torch import nn
import numpy as np
import random

"""
2022-04-10 17:54:06
Author: Can Wang
Weights reg loss for NeRF
"""
class WeightsRegLoss(nn.Module):
    def __init__(self, window=20):
        super(WeightsRegLoss, self).__init__()
        print('Init WeightsRegLoss Loss ')

        self.loss = torch.nn.MSELoss(reduction='mean')
        self.window = window

    def forward(self, weights):
        #print ("weights: ", weights.shape)  # fine: [518400, 192]
        W, H = 540, 960
        weights = weights.view(H, W, 192)

        x = random.randint(0, H-self.window-1)
        y = random.randint(0, W-self.window-1)

        total_loss = 0
        for i in range(self.window-1):
            for j in range(self.window-1):
                pos_x = x + i
                pos_y = y + j

                w1 = weights[pos_x, pos_y, :]
                w2 = weights[pos_x+1, pos_y, :]
                w3 = weights[pos_x, pos_y+1, :]

                loss = self.loss(w1, w2) + self.loss(w1, w3)
                total_loss += loss

        return total_loss
