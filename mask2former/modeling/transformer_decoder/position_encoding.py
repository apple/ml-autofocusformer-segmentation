# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
# Adapted for AutoFocusFormer by Ziwen 2023

"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn


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

    def forward(self, pos):
        '''
        pos - b x n x d
        '''
        b, n, d = pos.shape
        y_embed = pos[:, :, 1]  # b x n
        x_embed = pos[:, :, 0]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed.max() + eps) * self.scale
            x_embed = x_embed / (x_embed.max() + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)  # npf
        dim_t = self.temperature ** (2 * (dim_t.div(2, rounding_mode='floor')) / self.num_pos_feats)  # npf

        pos_x = x_embed[:, :, None] / dim_t  # b x n x npf
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.cat(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2
        )
        pos_y = torch.cat(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=2
        )
        pos = torch.cat((pos_x, pos_y), dim=2)  # b x n x d'
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
