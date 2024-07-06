from .recog_loss import MarginCosineProductLoss
from .age_loss import AgeLoss_DLDLV2
from .biattr_loss import CEL_Sigmoid
from .affect_loss import MyCrossEntropyLoss
from .parsing_loss import ParsingLoss
from .align_loss import AlignLoss

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def loss_entry(config):
    return globals()[config['type']](**config['kwargs'])




class LossesHolder(nn.Module):
    def __init__(self, tasks_cfg):
        super().__init__()
        self.task_cfg = tasks_cfg
        self.weights = dict()
        self.losses = nn.ModuleDict()

        self.weight_sum = 0.

        for i in range(len(self.task_cfg)):
            self.losses[self.task_cfg[i].name] = loss_entry(self.task_cfg[i].loss)
            self.weights[self.task_cfg[i].name] = self.task_cfg[i].loss_weight

            self.weight_sum += float(self.task_cfg[i].loss_weight)

    def forward(self, inputs):

        outputs = dict()
        total_loss = 0

        

        for task_name in inputs.keys():

            output = self.losses[task_name](inputs[task_name])
            outputs[task_name] = output

            total_loss += output["tloss"]*self.weights[task_name]

        total_loss /= self.weight_sum

        return total_loss, outputs
