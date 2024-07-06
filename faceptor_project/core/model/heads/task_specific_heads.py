

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cbam import ChannelGate
from typing import Tuple, Union, List, Optional
from ..geometry import heatmap2points

def head_entry(config):
    return globals()[config['type']](**config['kwargs'])
        

class TaskSpecificHeadsHolder(nn.Module):
    def __init__(self, subnets_cfg):
        super().__init__()
        self.subnets_cfg = subnets_cfg
        self.num_subnets = len(subnets_cfg)
        self.heads = nn.ModuleDict()

        for each in self.subnets_cfg.keys():
            head = head_entry(self.subnets_cfg[each])
            self.heads[each] = head

    def forward(self, inputs):

        for task_name in inputs.keys():
            inputs[task_name] = self.heads[task_name.split("_")[0]](inputs[task_name])
        
        return inputs


def select_features(features, indices=[3, 5, 7, 11]):
    return [features[ind] for ind in indices]

class RecognitionHead(nn.Module):
    def __init__(self, task_name="recog_", indices=[11], in_features=768, out_features=512):
        super().__init__()

        self.task_name = task_name
        self.indices = indices
        self.norm = nn.LayerNorm(in_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features, bias=False),
            nn.BatchNorm1d(num_features=in_features, eps=2e-5),
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            nn.BatchNorm1d(num_features=out_features, eps=2e-5)
        )

    def forward(self, inputs):

        features = inputs["backbone_output"]
        features = select_features(features, self.indices)[0] # B, D, H, W

        x = features.flatten(2).transpose(1, 2) # B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        outputs = {}
        outputs["head_output"] = x
        outputs.update(inputs)

        return outputs
    

class SwinFaceHead(nn.Module):
    def __init__(self, outputs, task_names, kernel_size=3, indices=[3, 5, 9, 11], 
                 in_features=768, out_features=512, drop_rate=0.5):
        super().__init__()
        self.indices = indices

        self.conv = nn.ModuleList(
            [nn.Conv2d(in_features, 
                       out_features//4, 
                       kernel_size, 
                       stride=1, 
                       padding=(kernel_size-1)//2, 
                       bias=False) for i in range(4)])

        self.channel_attention = ChannelGate(gate_channels=out_features)

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.norm = nn.BatchNorm1d(num_features=out_features, eps=2e-5)

        self.feature = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(out_features, out_features),
            nn.ReLU(True),
            nn.Dropout(drop_rate),)
        
        self.fcs = nn.ModuleDict()
        for i in range(len(outputs)):
            fc = nn.Linear(out_features, outputs[i])
            self.fcs[task_names[i]] = fc


    def forward(self, inputs):

        features = inputs["backbone_output"]

        features = select_features(features, self.indices) # 4 B, D, H, W

        x = [self.conv[i](features[i]) for i in range(4)]
        x = torch.cat(x, dim=1) # B, 512, H, W

        x = self.channel_attention(x)

        x = self.act(x)
        B, C, _, __ = x.shape
        x = self.pool(x).reshape(B, C) # B 512
        x = self.norm(x)
        x = self.feature(x)

        x = self.fcs[inputs["task_name"]](x) 

        outputs = {}
        outputs["head_output"] = x
        outputs.update(inputs)

        return outputs


class Activation(nn.Module):
    def __init__(self, name: Optional[str], **kwargs):
        super().__init__()
        if name == 'relu':
            self.fn = F.relu
        elif name == 'softplus':
            self.fn = F.softplus
        elif name == 'gelu':
            self.fn = F.gelu
        elif name == 'sigmoid':
            self.fn = torch.sigmoid
        elif name == 'sigmoid_x':
            self.epsilon = kwargs.get('epsilon', 1e-3)
            self.fn = lambda x: torch.clamp(
                x.sigmoid() * (1.0 + self.epsilon*2.0) - self.epsilon,
                min=0.0, max=1.0)
        elif name == None:
            self.fn = lambda x: x
        else:
            raise RuntimeError(f'Unknown activation name: {name}')

    def forward(self, x):
        return self.fn(x)
    


class ParsingOutput(nn.Module):
    def __init__(self, out_size=[512, 512]):
        super().__init__()
        self.out_size = out_size

    def forward(self, out):
        parsingmap = F.interpolate(out, size=self.out_size, mode='bilinear', align_corners=False) # B N H, W
        parsingmap = parsingmap.permute(0, 2, 3, 1)

        return parsingmap

class AlignOutput(nn.Module):
    def __init__(self, heatmap_act='sigmoid') -> None:
        super().__init__()
        self.heatmap_act = Activation(heatmap_act)

    def forward(self, out):

        heatmap_acted = self.heatmap_act(out)

        landmark = heatmap2points(heatmap_acted)
        return {"landmark": landmark, 'heatmap': out, 'heatmap_acted': heatmap_acted}


def output_entry(out_type):
    if out_type == "None":
        return None

    return globals()[out_type]()


def _make_fpns(vision_patch_size: int, output_channels: int):
    if vision_patch_size in {16, 14}:
        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels,
                               kernel_size=2, stride=2),
            nn.SyncBatchNorm(output_channels),
            nn.GELU(),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2))

        fpn2 = nn.ConvTranspose2d(
            output_channels, output_channels, kernel_size=2, stride=2)
        fpn3 = nn.Identity()
        fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.ModuleList([fpn1, fpn2, fpn3, fpn4])
    elif vision_patch_size == 8:
        fpn1 = nn.Sequential(nn.ConvTranspose2d(
            output_channels, output_channels, kernel_size=2, stride=2))
        fpn2 = nn.Identity()
        fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)
        fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)
        return nn.ModuleList([fpn1, fpn2, fpn3, fpn4])
    else:
        raise NotImplementedError()

class MMSEG_UPerHead(nn.Module):
    """Wraps the UPerHead from mmseg for segmentation.
    """

    def __init__(self, outputs, task_names, out_types, indices=[3, 5, 9, 11],
                 in_features=768, channels=512, out_size=[512, 512]):
        super().__init__()


        self.outputs = outputs
        self.task_names = task_names
        self.out_types = out_types
        self.indices = indices
        
        self.out_ranges = dict()
        self.out_modules = nn.ModuleDict()
        out_sum = 0

        for i in range(len(outputs)):
            
            self.out_ranges[self.task_names[i]] = [out_sum, out_sum + self.outputs[i]]
            out_sum += self.outputs[i]

            out_module = output_entry(self.out_types[i])
            self.out_modules[task_names[i]] = out_module

        self.fpns = _make_fpns(16, in_features)

        from mmseg.models.decode_heads import UPerHead
        self.head = UPerHead(
            in_channels=[in_features for i in range(4)],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=channels,
            dropout_ratio=0.1,
            num_classes=sum(outputs),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
            

    def forward(self, inputs):

        features = inputs["backbone_output"]
        features = select_features(features, self.indices) # 4 B, D, H, W

        for i, fpn in enumerate(self.fpns):
            features[i] = fpn(features[i])

        output = self.head(features) # B, N, S, S

        temp = self.out_ranges[inputs["task_name"]]
        x = output[:, temp[0]:temp[1], :, :]

        x = self.out_modules[inputs["task_name"]](x)

        outputs = {}
        outputs["head_output"] = x
        outputs.update(inputs)

        return outputs

