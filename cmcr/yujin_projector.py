import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MLP_Half(nn.Module):
    def __init__(self, channel, res_expansion=2.0, bias=True, activation='relu'):
        super().__init__()
        expanded = int(channel * res_expansion)
        self.net = nn.Sequential(
            nn.Linear(channel, expanded, bias=bias),
            nn.BatchNorm1d(expanded),
            get_activation(activation)
        )
        self.out_proj = nn.Linear(expanded, channel)

    def forward(self, x):
        x = self.net(x)
        x = self.out_proj(x)
        return x


class TriAlignHead(nn.Module):
    def __init__(
        self,
        text_input_dim=768,     # XLM-R output dim
        image_input_dim=512,    # CLIP ViT-B/16 output dim
        proj_dim=512,           # Shared embedding space dim
        res_expansion=2.0,
        activation='relu'
    ):
        super().__init__()
        # Projector for multilingual text (한국어 / 영어)
        self.text_proj = MLP_Half(channel=text_input_dim, res_expansion=res_expansion, activation=activation)
        self.text_out = nn.Linear(text_input_dim, proj_dim)

        # Projector for image embedding
        self.image_proj = MLP_Half(channel=image_input_dim, res_expansion=res_expansion, activation=activation)
        self.image_out = nn.Linear(image_input_dim, proj_dim)

    def forward_text(self, x):
        """
        Input: x - (batch_size, text_input_dim)
        Output: (batch_size, proj_dim), normalized
        """
        x = self.text_proj(x)
        x = self.text_out(x)
        return F.normalize(x, dim=-1)

    def forward_image(self, x):
        """
        Input: x - (batch_size, image_input_dim)
        Output: (batch_size, proj_dim), normalized
        """
        x = self.image_proj(x)
        x = self.image_out(x)
        return F.normalize(x, dim=-1)

    def get_device(self):
        return next(self.parameters()).device

    def init_weights(self, mode='xav'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == 'eye':
                    if m.weight.size(0) == m.weight.size(1):
                        nn.init.eye_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                elif mode == 'xav':
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
