


import torch
import math
import torch.nn as nn
from core.utils import accuracy
import torch.nn.functional as F
import audtorch
import numpy as np

from core.model.geometry import normalize_points, points2heatmap


class AlignLoss(nn.Module):
    def __init__(self, input_size=512, heatmap_size=128, heatmap_radius=5.0):
        super(AlignLoss, self).__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.heatmap_radius = heatmap_radius

    def forward(self, inputs):

        head_output = inputs["head_output"]
        label = inputs["label"]

        pred_landmark = head_output["landmark"]
        landmark = normalize_points(label, self.input_size, self.input_size)
        

        coord_l1_loss = (landmark - pred_landmark).norm(dim=-1).mean()

        pred_heatmap = F.interpolate(head_output['heatmap'], (self.heatmap_size, self.heatmap_size),
                                     mode='bilinear', align_corners=False)
        

        heatmap = points2heatmap(landmark, (self.heatmap_size, self.heatmap_size), self.heatmap_radius)

        bce_loss = F.binary_cross_entropy_with_logits(pred_heatmap, heatmap, reduction='none').mean()

        loss = coord_l1_loss + bce_loss
    
        return {'task_name': inputs["task_name"], 
                'tloss': loss, 
                'top1': np.array(0.0), 
                'losses': [coord_l1_loss, bce_loss], 
                'weights': [1.0, 1.0], 
                'loss_names':["coord_l1_loss", "heatmap_ce_loss"]}

    def __repr__(self):
        return self.__class__.__name__