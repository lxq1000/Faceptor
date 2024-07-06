import torch
import math
import torch.nn as nn
from core.utils import accuracy
import torch.nn.functional as F
import audtorch


class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):

        logits = inputs["head_output"]
        label = inputs["label"]

        loss = self.ce(logits, label)
        top1 = accuracy(logits.data, label.cuda(), topk=(1,))[0]
    
        return {'task_name': inputs["task_name"], 
                'tloss': loss, 
                'top1': top1, 
                'losses': [loss], 
                'weights': [1.0], 
                'loss_names':['ce']}

    def __repr__(self):
        return self.__class__.__name__
    