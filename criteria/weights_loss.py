import os
import torch
from torch import nn
import numpy as np
import random

"""
2022-03-14 15:02:23 
Author: Can Wang
Weights loss for each ray of NeRF
Can directly use Eikonal loss for NeuS and VolSDF
"""
class WeightsLoss(nn.Module):
    def __init__(self, steps=1, if_tranc=False, tranc=50):
        super(WeightsLoss, self).__init__()
        print('Init WeightsLoss Loss ')

        self.init_eta = 0.88
        self.decay = 0.01
        self.loss = torch.nn.MSELoss(reduction='mean')

        self.if_tranc = if_tranc  # whether tranc the ray
        self.tranc = tranc  # tranc the first 50 points
        self.steps = steps
        if self.if_tranc:
            self.start = self.tranc
        else:
            self.start = 0  # sample point start
        self.end = 192  # sample point end
        self.mask_split = 180

    def forward(self, weights, deltas, mask=None, use_mask=False):
        # transmittance (XXX, 128 + 64)
        #print ("weights: ", weights.shape)  # fine: [537600, 192]
        #print ("deltas: ", deltas.shape)  # fine: [537600, 192]

        if use_mask: # use mask 
            vals_x1 = range(self.start, self.mask_split)
            vals_x2 = range(self.mask_split, self.end)
            x1_mask = random.sample(vals_x1, 1)[0]
            x2_mask = random.sample(vals_x2, 1)[0] 
            #print ("mask x1:x2, %d, %d" % (x1_mask, x2_mask))
            distance_mask = torch.sum(deltas[:, x1_mask: x2_mask], dim=1)  # [537600]
            w1_mask = weights[:, x1_mask]
            w2_mask = weights[:, x2_mask]
            loss_mask = torch.mean(w1_mask * w2_mask * distance_mask * (1-mask)) # only for background 
            #print (loss_mask)
            
        vals = range(self.start, self.end)
        x1, x2 = sorted(random.sample(vals, 2))
        #print ("x1:x2, %d, %d" % (x1, x2))
        distance = torch.sum(deltas[:, x1: x2], dim=1)  # [537600]
        w1 = weights[:, x1]
        w2 = weights[:, x2]
        loss = torch.mean(w1 * w2 * distance)

        if use_mask:
            loss = loss + 10 * loss_mask 

        """
        self.init_eta = self.init_eta - self.decay * epoch
        target = torch.tensor(self.init_eta).cuda()
        mean_trans = torch.mean(transmittance, dim=1) # XXX
        loss = self.loss(mean_trans, target)
        """
        return loss
