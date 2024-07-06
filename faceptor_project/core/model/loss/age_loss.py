


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class AgeLoss_DLDLV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.Sigmoid()

    
    def kl_loss(self, inputs, labels):
        criterion = nn.KLDivLoss(reduce=False)
        outputs = torch.log(inputs)
        loss = criterion(outputs, labels)
        loss = loss.sum()/loss.shape[0]
        return loss
    
    def l1_loss(self, inputs, labels):
        criterion = nn.L1Loss(reduction='mean')
        loss = criterion(inputs, labels.float())
        return loss

    def forward(self, inputs):

        outputs = inputs["head_output"]

        outputs = self.act(outputs)
        outputs = F.normalize(outputs, p=1, dim=1)


        label = inputs["label"]

        dis = label["distribution"]
        age = label["avg_label"]

        rank = torch.Tensor([i for i in range(101)]).cuda()
        ages = torch.sum(outputs*rank, dim=1)


        kl_loss = self.kl_loss(outputs, dis)
        l1_loss = self.l1_loss(ages, age)
        tloss = kl_loss + l1_loss

        top1 = np.array(abs(ages.detach().cpu().numpy() - age.cpu().numpy()) <= 5, dtype=np.float32).mean()*100
    
        return {'task_name': inputs["task_name"], 
                'tloss': tloss, 
                'top1': top1, 
                'losses': [kl_loss, l1_loss], 
                'weights': [1.0, 1.0], 
                'loss_names':['kl_loss', 'l1_loss']}

    def __repr__(self):
        return self.__class__.__name__