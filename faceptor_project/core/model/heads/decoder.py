

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cbam import ChannelGate
from typing import Tuple, Union, List, Optional
from ..geometry import heatmap2points


import logging

from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import math

from core.model.heads.task_specific_heads import output_entry, _make_fpns, select_features


def decoder_entry(decoder_type):
    return globals()[decoder_type]


class DecoderNewHolder(nn.Module):
    def __init__(self,
                 task_names=["recog_ms1mv3", "age_morph2", "biattr_celeba", "affect_affectnet", "affect_rafdb", "parsing_celebam"],
                 query_nums=[2, 101, 40, 7, 7, 19],
                 interpreter_types=["feature_v2", "value", "value", "value", "value", "map"],
                 out_types=[None, None, None, None, None, "ParsingOutput"],
                 hidden_dim=256,
                 decoder_type="TransformerDecoder",
                 levels=[i for i in range(12)],
                 new_levels=[1, 3, 5, 7, 9]):
        super().__init__()

        self.task_names = task_names

        if task_names[0].split("_")[0] == "recog":
            self.decoder = decoder_entry(decoder_type)(task_names=task_names[1:], query_nums=query_nums[1:], levels=levels, new_levels=new_levels)
        else:
            self.decoder = decoder_entry(decoder_type)(task_names=task_names, query_nums=query_nums, levels=levels, new_levels=new_levels)

        self.interpreter_dict = {task_names[i]: interpreter_types[i] for i in range(len(task_names))}
        self.interpreters = nn.ModuleDict()

        if "feature_v2" in interpreter_types:
            self.interpreters["feature_v2"] = FeatureInterpreter_v2()
        if "feature" in interpreter_types:
            self.interpreters["feature"] = FeatureInterpreter() #

        if "value" in interpreter_types:
            self.interpreters["value"] = ValueInterpreter(hidden_dim)
        if "map" in interpreter_types:
            self.interpreters["map"] = MapInterpreter(hidden_dim)
        if "map_v2" in interpreter_types:
            self.interpreters["map_v2"] = MapInterpreter_V2(hidden_dim)


        self.out_modules = nn.ModuleDict()

        for i in range(len(task_names)):
            out_module = output_entry(out_types[i])
            self.out_modules[task_names[i]] = out_module
    

    def forward(self, inputs):

        for task_name in inputs.keys():
            if str(task_name).split("_")[0] == "recog":
                pass
            else:
                inputs[task_name] = self.decoder(inputs[task_name])

            inputs[task_name] = self.interpreters[self.interpreter_dict[task_name]](inputs[task_name])

            if self.out_modules[task_name]:
                inputs[task_name]["head_output"] = self.out_modules[task_name](inputs[task_name]["head_output"])
        
        return inputs
    

class FeatureInterpreter(nn.Module):
    def __init__(self, hidden_dim=512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
            nn.BatchNorm1d(num_features=hidden_dim, eps=2e-5),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
            nn.BatchNorm1d(num_features=hidden_dim, eps=2e-5)
        )

    def forward(self, inputs):
        x = inputs["decoder_output"].permute(1, 2, 0) # B C 2
        x = x.flatten(1) # B 512
        x = self.head(x)

        inputs["head_output"] = x
        return inputs

class FeatureInterpreter_v2(nn.Module):
    def __init__(self, in_features=768, out_features=512) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(in_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features, bias=False),
            nn.BatchNorm1d(num_features=in_features, eps=2e-5),
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            nn.BatchNorm1d(num_features=out_features, eps=2e-5)
        )

    def forward(self, inputs):
        features = inputs["backbone_output"][-1] # B, D, H, W

        x = features.flatten(2).transpose(1, 2) # B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        inputs["head_output"] = x
        return inputs


class ValueInterpreter(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x = inputs["decoder_output"] # Q B C
        x = x.permute(1, 0, 2) # B Q C
        x = self.fc(x) # B Q 1
        x = x.flatten(1) # B Q

        inputs["head_output"] = x
        return inputs


class MapInterpreter(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()

        self.mask_map = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            Norm2d(hidden_dim),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
        )

    def forward(self, inputs):

        src = inputs["decoder_src"] # HW B C

        hw, b, c = src.shape
        h = w = int(math.sqrt(hw))
        src = src.view(h, w, b, c).permute(2, 3, 0, 1) #B C H W 
        src = self.mask_map(src) # B C 128 128
        b, c, h, w = src.shape
        src = src.flatten(2) # B, C, HW

        output = inputs["decoder_output"] # Q B C
        output = output.permute(1, 0, 2) # B Q C
        b, q, c = output.shape

        x = torch.bmm(output, src) # B Q 128*128

        x = x.view(b, q, h, w)   

        inputs["head_output"] = x
        return inputs


class MapInterpreter_V2(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()

        self.mask_map = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            Norm2d(hidden_dim),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
        )

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward(self, inputs):

        src = inputs["decoder_src"] # HW B C

        # from backbone
        hw, b, c = src.shape
        h = w = int(math.sqrt(hw))
        src = src.view(h, w, b, c).permute(2, 3, 0, 1) #B C H W 
        mask_features = self.mask_map(src) # B C H W


        decoder_output = inputs["decoder_output"] # Q B C
        decoder_output = decoder_output.permute(1, 0, 2) # B Q C
        mask_embed = self.mask_embed(decoder_output) # B Q C

        
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        inputs["head_output"] = outputs_mask

        return inputs


class Norm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        self.normalized_shape = (embed_dim,)

        #  >>> workaround for compatability
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln.weight = self.weight
        self.ln.bias = self.bias

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_post_deep(self, tgt,
                         tgt_mask: Optional[Tensor] = None,
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt * (3 * self.net_depth) ** 0.25 + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.arch == 'pre_norm':
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        elif self.arch == 'deepnorm':
            return self.forward_post_deep(tgt, tgt_mask,
                                          tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self.memory = None # For Attention 

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        

        self.memory = attn.clone()

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.arch == 'pre_norm':
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        elif self.arch == 'deepnorm':
            raise NotImplementedError
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", arch=False, net_depth=9):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'linear' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_post_deep(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt * (3 * self.net_depth) ** 0.25 + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.arch == 'pre_norm':  # false
            return self.forward_pre(tgt)
        elif self.arch == 'deepnorm':
            return self.forward_post_deep(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def custom_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res


def weights_init(module):
    """ Initialize the weights, copy from CTran"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class TransformerDecoderLevel(nn.Module):
    def __init__(self,
                 in_channels=768,
                 hidden_dim=256,
                 task_names=["recog_ms1mv3", "age_morph2", "biattr_celeba", "affect_affectnet", "affect_rafdb", "parsing_celebam", "align_300w"],
                 query_nums=[2, 101, 40, 7, 7, 19],
                 nheads=8,
                 dim_feedforward=2048,
                 num_layers=9,
                 enforce_input_project=False,  # False
                 levels=[i for i in range(12)],
                 arch="fan_in",
                 pre_norm=False,  # False
                 cross_pos_embed="anchor",  # use pos embed in cross attention layers
                 **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.cross_pos_embed = cross_pos_embed

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)


        self.query_feat = nn.ModuleDict()
        self.query_embed = nn.ModuleDict()

        for i in range(len(task_names)):
            task = task_names[i]
            num_q = query_nums[i]
            query_feat_temp = nn.Embedding(num_q, hidden_dim)
            query_embed_temp = nn.Embedding(num_q, hidden_dim)

            self.query_feat[task]=query_feat_temp
            self.query_embed[task]=query_embed_temp


        # level embedding (originally 3 scales)
        self.levels = levels
        self.num_feature_levels = len(levels)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # define Transformer decoder here
        self.hidden_dim = hidden_dim
        self.num_heads = nheads
        self.num_layers = num_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            )

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ) if self.cross_pos_embed == 'anchor' else None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.adapt_pos2d is not None:
            for each in self.query_embed.keys():
                nn.init.uniform_(self.query_embed[each].weight.data, 0, 1)

    def get_vis_token_pos_embed(self, shape=None, batch=1):
        if self.cross_pos_embed == 'anchor':
            assert shape is not None
            mask = torch.zeros(batch, *shape, dtype=torch.bool).cuda()
            H_n, W_n = shape
            pos_col, pos_row = mask2pos(mask)  # (1xh, 1xw) workaround to utilize existing codebase
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, H_n, 1).unsqueeze(-1),
                                pos_col.unsqueeze(2).repeat(1, 1, W_n).unsqueeze(-1)], dim=-1)  # 1xhxwx2
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d, self.hidden_dim // 2))  # 1xhxwxc
            return posemb_2d.flatten(1,2).permute(1, 0, 2)  # BxHWxC -> HWxBxC
        else:
            raise NotImplementedError(f"unknown self.cross_pos_embed: {self.cross_pos_embed}")

    def forward(self, input):

        x = input["backbone_output"]

        src = []
        pos = []

        for j in range(self.num_feature_levels):

            src_temp = self.input_proj[j](x[self.levels[j]]).flatten(2).permute(2, 0, 1) # H*W B D(256)

            loc_emb = self.get_vis_token_pos_embed(x[self.levels[j]].shape[-2:]).cuda()
            level_emb = self.level_embed.weight[j]
            pos_temp = loc_emb + level_emb[None, None, :]

            src.append(src_temp)
            pos.append(pos_temp)

            if self.levels[j] == 11:
                decoder_src = src_temp

        src = torch.cat(src, dim=0) # 6*H*W B D
        pos = torch.cat(pos, dim=0)


        _, bs, _ = src.shape

        task_name = input["task_name"]

        # QxNxC
        query_embed = self.query_embed[task_name].weight.unsqueeze(1).repeat(1, bs, 1).cuda()
        output = self.query_feat[task_name].weight.unsqueeze(1).repeat(1, bs, 1).cuda()

        for i in range(self.num_layers):

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos,
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

        input["decoder_output"] = output # Q B C
        input["decoder_src"] = decoder_src # HW B C

        return input
    

class TransformerDecoderLevelAddLevel(nn.Module):
    def __init__(self,
                 in_channels=768,
                 hidden_dim=256,
                 task_names=["recog_ms1mv3", "age_morph2", "biattr_celeba", "affect_affectnet", "affect_rafdb", "parsing_celebam", "align_300w"],
                 query_nums=[2, 101, 40, 7, 7, 19],
                 #share_token=50,
                 nheads=8,
                 dim_feedforward=2048,
                 num_layers=9,
                 enforce_input_project=False,  # False
                 levels=[11],
                 new_levels=[1, 3, 5, 7, 9],
                 arch="fan_in",
                 pre_norm=False,  # False
                 cross_pos_embed="anchor",  # use pos embed in cross attention layers
                 **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.cross_pos_embed = cross_pos_embed

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)


        self.query_feat = nn.ModuleDict()
        self.query_embed = nn.ModuleDict()

        #self.share_feat = nn.Embedding(share_token, hidden_dim)
        #self.share_embed = nn.Embedding(share_token, hidden_dim)

        for i in range(len(task_names)):
            task = task_names[i]
            num_q = query_nums[i]
            query_feat_temp = nn.Embedding(num_q, hidden_dim)
            query_embed_temp = nn.Embedding(num_q, hidden_dim)

            self.query_feat[task]=query_feat_temp
            self.query_embed[task]=query_embed_temp


        # level embedding (originally 3 scales)
        self.levels = levels
        self.num_feature_levels = len(levels)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.new_levels = new_levels
        self.new_num_feature_levels = len(new_levels)
        self.new_level_embed = nn.Embedding(self.new_num_feature_levels, hidden_dim)
        self.new_input_proj = nn.ModuleList()
        for _ in range(self.new_num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.new_input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.new_input_proj[-1])
            else:
                self.new_input_proj.append(nn.Sequential())

        

        # define Transformer decoder here
        self.hidden_dim = hidden_dim
        self.num_heads = nheads
        self.num_layers = num_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            )

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ) if self.cross_pos_embed == 'anchor' else None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.adapt_pos2d is not None:
            for each in self.query_embed.keys():
                nn.init.uniform_(self.query_embed[each].weight.data, 0, 1)

    def get_vis_token_pos_embed(self, shape=None, batch=1):
        if self.cross_pos_embed == 'anchor':
            assert shape is not None
            mask = torch.zeros(batch, *shape, dtype=torch.bool).cuda()
            H_n, W_n = shape
            pos_col, pos_row = mask2pos(mask)  # (1xh, 1xw) workaround to utilize existing codebase
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, H_n, 1).unsqueeze(-1),
                                pos_col.unsqueeze(2).repeat(1, 1, W_n).unsqueeze(-1)], dim=-1)  # 1xhxwx2
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d, self.hidden_dim // 2))  # 1xhxwxc
            return posemb_2d.flatten(1,2).permute(1, 0, 2)  # BxHWxC -> HWxBxC
        else:
            raise NotImplementedError(f"unknown self.cross_pos_embed: {self.cross_pos_embed}")

    def forward(self, input):

        x = input["backbone_output"]

        src = []
        pos = []

        for j in range(self.num_feature_levels):

            src_temp = self.input_proj[j](x[self.levels[j]]).flatten(2).permute(2, 0, 1)

            loc_emb = self.get_vis_token_pos_embed(x[self.levels[j]].shape[-2:]).cuda()
            level_emb = self.level_embed.weight[j]
            pos_temp = loc_emb + level_emb[None, None, :]

            src.append(src_temp)
            pos.append(pos_temp)

            if self.levels[j] == 11:
                decoder_src = src_temp

        for j in range(self.new_num_feature_levels):

            src_temp = self.new_input_proj[j](x[self.new_levels[j]]).flatten(2).permute(2, 0, 1)

            loc_emb = self.get_vis_token_pos_embed(x[self.new_levels[j]].shape[-2:]).cuda()
            level_emb = self.new_level_embed.weight[j]
            pos_temp = loc_emb + level_emb[None, None, :]

            src.append(src_temp)
            pos.append(pos_temp)

        src = torch.cat(src, dim=0)
        pos = torch.cat(pos, dim=0)

        _, bs, _ = src.shape

        task_name = input["task_name"]

        task_type = str(task_name).split("_")[0]

        # QxNxC
        query_embed = self.query_embed[task_name].weight.unsqueeze(1).repeat(1, bs, 1).cuda()
        output = self.query_feat[task_name].weight.unsqueeze(1).repeat(1, bs, 1).cuda()


        for i in range(self.num_layers):

            # >>> ** cross-attn **  attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos,
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

        input["decoder_output"] = output # Q B C
        input["decoder_src"] = decoder_src # HW B C

        return input


# 工具类和函数：

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    # QxBx2
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t  # QxBx128
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)

    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed
