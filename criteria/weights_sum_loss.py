import os
import torch
from torch import nn
import numpy as np
import random

"""
2022-04-08 21:55:23
Author: Can Wang
Weights sum loss for each ray of NeRF
"""
class WeightsSumLoss(nn.Module):
    def __init__(self, steps=20, if_tranc=True, tranc=50):
        super(WeightsSumLoss, self).__init__()
        print('Init WeightsSumLoss Loss ')

        self.loss = torch.nn.MSELoss(reduction='mean')

        self.if_tranc = if_tranc  # whether tranc the ray
        self.tranc = tranc  # tranc the first 50 points
        if self.if_tranc:
            self.start = self.tranc
        else:
            self.start = 0  # sample point start
        self.end = 192  # sample point end
        self.mask_split = 185

        self.steps = 20  # sample steps
        self.first = True
        self.avg = None  # [N]

    def forward(self, weights):
        # transmittance (XXX, 128 + 64)
        #print ("weights: ", weights.shape)  # fine: [537600, 192]
        #print ("deltas: ", deltas.shape)  # fine: [537600, 192]
        if self.first:
            weights_tranc = weights[:, self.start: self.mask_split]  # [N, XXX]
            self.avg = torch.mean(weights_tranc, dim=1)  # [N]
            print ("Avg is: ", self.avg)
            self.first = False 

        vals_x1 = range(self.start, self.mask_split)
        vals_x2 = range(self.mask_split, self.end)

        total_loss = 0
        for _ in range(self.steps):
            x1 = random.sample(vals_x1, 1)[0]
            x2 = random.sample(vals_x2, 1)[0]
            w1 = weights[:, x1]
            w2 = weights[:, x2]
            loss = self.loss(w1, self.avg) - self.loss(w2, w1)
            total_loss += loss

        return total_loss
