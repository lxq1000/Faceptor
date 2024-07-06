import torch
import math
import torch.nn as nn
from core.utils import accuracy


class MarginCosineProductLoss(nn.Module):
    def __init__(self, in_features, out_features, scale, margin, with_theta=False, label_smooth=-1):
        super(MarginCosineProductLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.with_theta = with_theta
        self.thetas = []
        self.classifier = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.ce = torch.nn.CrossEntropyLoss()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)

    def forward(self, inputs):

        input = inputs["head_output"]
        label = inputs["label"]

        cosine = self.cosine_sim(input, self.classifier)
        thetas = [math.acos(cosine[i, int(label[i])].item()) / math.pi * 180 for i in range(cosine.size(0))]
        self.thetas.append(thetas)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.scale * (cosine - one_hot * self.margin)
        loss = self.ce(output, label)
        top1 = accuracy(output.data, label.cuda(), topk=(1, 5))[0]
        
        return {'task_name': inputs["task_name"], 
                'tloss': loss, 
                'top1': top1, 
                'losses': [loss], 
                'weights': [1.0], 
                'loss_names':['cosface']}

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', scale=' + str(self.scale) \
            + ', margin='+ str(self.margin) + ')'