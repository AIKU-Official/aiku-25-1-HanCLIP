import os
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

# from clap_src.clap_module import create_model
# from clap_src.training.data import get_audio_features
# from clap_src.training.data import int16_to_float32, float32_to_int16

# from transformers import RobertaTokenizer
# from clap_src.clap_module.factory import load_state_dict
# from clap_src.clap_module.htsat import create_htsat_model
# from clap_src.clap_module.model import CLAPAudioCfp
import copy

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU(inplace=True)
def get_clones(module: nn.Module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    def __init__(
        self, channel=512, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(channel, int(channel * res_expansion), bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(channel * res_expansion), channel, bias=bias),
        )

    def forward(self, x):
        return self.net2(self.net1(x))


# class MLP_Half(nn.Module):
#     def __init__(
#         self, channel=512, res_expansion=1.0, bias=True, activation='relu'):
#         super().__init__()
#         self.act = get_activation(activation)
#         self.net = nn.Sequential(
#             nn.Linear(channel, int(channel * res_expansion), bias=bias),
#             nn.BatchNorm1d(int(channel * res_expansion)),
#             self.act
#         )

#     def forward(self, x):
#         return self.net(x)


# class CLAPCLIP_Head(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Head_A = MLP(res_expansion=2.0)
#         self.Head_B = MLP(res_expansion=2.0)

#     def get_device(self):
#         return next(self.parameters()).device
    
#     def init_weights(self, mode):
#         # initialize transformer
#         if mode == 'eye':
#             for m in self.parameters():
#                 if m.dim() > 1:
#                     nn.init.eye_(m)
#         elif mode == 'xav':
#             for m in self.parameters():
#                 if m.dim() > 1:
#                     nn.init.xavier_uniform_(m)
    
# class ULIPCLIP_Head(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Head_A = MLP_Half()
#         self.Head_B = MLP_Half()

#     def get_device(self):
#         return next(self.parameters()).device
    
#     def init_weights(self, mode):
#         # initialize transformer
#         if mode == 'eye':
#             for m in self.parameters():
#                 if m.dim() > 1:
#                     nn.init.eye_(m)
#         elif mode == 'xav':
#             for m in self.parameters():
#                 if m.dim() > 1:
#                     nn.init.xavier_uniform_(m)

class HanCLIP_Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model == 'minilm':
            self.Head_A = nn.Sequential(
                nn.Linear(384, 384 * 2),
                nn.BatchNorm1d(384*2),
                nn.ReLU(),
                nn.Linear(384*2, 512),
            )
        elif model == 'xlmr':
            self.Head_A = nn.Sequential(
                nn.Linear(768, 768 * 2),
                nn.BatchNorm1d(768*2),
                nn.ReLU(),
                nn.Linear(768*2, 512),
            )
        elif model == 'e5':
            self.Head_A = nn.Sequential(
                nn.Linear(768, 768*2),
                nn.BatchNorm1d(768*2),
                nn.ReLU(),
                nn.Linear(768*2, 512),
            )
            
        self.Head_B = nn.Sequential(
            nn.Linear(512, 512 * 2),
            nn.BatchNorm1d(512*2),
            nn.ReLU(),
            nn.Linear(512*2, 512),
        )

    def forward_text(self, x):
        """
        Input: x - (batch_size, text_input_dim)
        Output: (batch_size, proj_dim), normalized
        """
        x = self.Head_A(x)
        return F.normalize(x, dim=-1)