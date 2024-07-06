


import torch
import math
import torch.nn as nn
from core.utils import accuracy
import torch.nn.functional as F
import audtorch
import numpy as np


class ParsingLoss(nn.Module):
    def __init__(self):
        super(ParsingLoss, self).__init__()

    def forward(self, inputs):

        logits = inputs["head_output"]

        batch, h, w, channels = logits.shape

        logits = logits.reshape(-1, channels) 
        label = inputs["label"].view(-1)
        loss = F.cross_entropy(logits, target=label, reduction='none').mean()        
    
        return {'task_name': inputs["task_name"], 
                'tloss': loss, 
                'top1': np.array(0.0), 
                'losses': [loss], 
                'weights': [1.0], 
                'loss_names':['ce']}

    def __repr__(self):
        return self.__class__.__name__