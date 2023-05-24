#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
from typing import Callable, Dict, List, Optional, Union
import math

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from ..backbone.point_utils import knn_keops, upsample_feature_shepard
from ..backbone.aff import pre_table, rel_pos_width, table_width
from ..clusten import CLUSTENWFFunction, MSDETRPCFunction


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


def scale_pos(last_pos, last_ss, cur_ss, no_bias=False):
    """
    Scales the positions from last_ss scale to cur_ss scale.
    Args:
        last_pos - ... x 2, 2D positions
        *_ss - (h,w), height and width
        no_bias - bool, if True, move the positions to the center of the grid and then scale,
                        so that there is no bias toward the upperleft corner
    Returns:
        res - ... x 2, scaled 2D positions
    """
    if last_ss[0] == cur_ss[0] and last_ss[1] == cur_ss[1]:
        return last_pos
    last_h, last_w = last_ss
    cur_h, cur_w = cur_ss
    h_ratio = cur_h / last_h
    w_ratio = cur_w / last_w
    ret = last_pos.clone()
    if no_bias:
        ret += 0.5
    ret[..., 0] *= w_ratio
    ret[..., 1] *= h_ratio
    if no_bias:
        ret -= 0.5
    return ret


class MSDeformAttnTransformerEncoderOnlyPc(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=3, enc_n_points=4,
                 shepard_power=3.0, shepard_power_learnable=True
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayerPc(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              shepard_power, shepard_power_learnable)
        self.encoder = MSDeformAttnTransformerEncoderPc(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttnPc):
                m._reset_parameters()
        normal_(self.level_embed)

    def forward(self, srcs, poss, spatial_shapes, pos_embeds, nb_idx):
        """
        Args:
            srcs - [b x n x c], a list of feature point clouds
            poss - [b x n x 2], a list of point cloud positions
            spatial_shapes - [(h,w)], a list of canvas sizes of the point clouds
            pos_embeds - [b x n x c], a list of positional embeddings
            nb_idx - [b x (h*w) x 4], a list of idx of nearest 4 neighbors for all positions in the finest resolution
        """
        lvl_pos_embeds = []
        for lvl, pos_embed in enumerate(pos_embeds):
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embeds.append(lvl_pos_embed)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)  # l x 2

        # encoder
        memory = self.encoder(srcs, poss, spatial_shapes, lvl_pos_embeds, nb_idx)

        return memory


class MSDeformAttnPc(nn.Module):
    def __init__(self, d_model, n_levels, n_heads, n_points, shepard_power, shepard_power_learnable):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        if shepard_power_learnable:
            self.shepard_power = nn.Parameter(shepard_power * torch.ones(1))
        else:
            self.shepard_power = shepard_power

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, querys, poss, values, spatial_shapes, nb_idx):
        """
        Args:
            querys, values - [b x n x c]
            poss - [b x n x 2]
            spatial_shapes - l x 2
            nb_idx - [b x (h*w) x 4]
        """
        b, _, c = querys[0].shape
        h = self.n_heads
        l = self.n_levels
        k = self.n_points
        c_ = c // h

        grid_hw = spatial_shapes[-1]

        values = self.value_proj(torch.cat(values, dim=1)).reshape(b, -1, h, c_).permute(0, 2, 1, 3).reshape(b*h, -1, c_)

        sampling_offsets = [self.sampling_offsets(query).view(b, -1, h, l, k, 2) for query in querys]  # b x n x h x l x k x 2
        attention_weights = [self.attention_weights(query).view(b, -1, h, l*k) for query in querys]  # b x n x h x l*k
        attention_weights = [F.softmax(attention_weight, -1).view(b, -1, h, l, k) for attention_weight in attention_weights]  # b x n x h x l x k
        scaled_poss = []
        for i, pos in enumerate(poss):
            scaled_pos = []
            for j in range(l):
                s_pos = scale_pos(pos, spatial_shapes[i], spatial_shapes[j], no_bias=True)
                scaled_pos.append(s_pos)
            scaled_pos = torch.stack(scaled_pos, dim=2)  # b x n x l x 2
            scaled_poss.append(scaled_pos)

        sampling_locations = [scaled_pos[:, :, None, :, None, :] + sampling_offset for scaled_pos, sampling_offset in zip(scaled_poss, sampling_offsets)]

        outputs = []

        for i in range(l):
            sampled_values = []
            nn_idxs = []
            idx_acc = 0
            nn_weights = []
            for j in range(l):
                sampling_location = sampling_locations[i][:, :, :, j].permute(0, 2, 1, 3, 4).reshape(b*h, -1, 2)  # b*h x n*k x 2

                # fetch nn_idx from lookup
                scaled_loc = scale_pos(sampling_location, spatial_shapes[j], grid_hw, no_bias=True)
                scaled_loc = scaled_loc.round().long()
                scaled_loc_x = scaled_loc[..., 0].clamp(0, grid_hw[1]-1)
                scaled_loc_y = scaled_loc[..., 1].clamp(0, grid_hw[0]-1) * grid_hw[1]
                gather_idx = scaled_loc_x + scaled_loc_y  # b*h x n*k
                nb_idx_real = nb_idx[j].gather(index=gather_idx.view(b, -1, 1).expand(-1, -1, 4), dim=1).reshape(b*h, -1, 4)

                nn_idxs.append(nb_idx_real + idx_acc)
                nn_weight = upsample_feature_shepard(sampling_location.contiguous(), poss[j].unsqueeze(1).expand(-1, h, -1, -1).reshape(b*h, -1, 2).contiguous(), None, power=self.shepard_power, custom_kernel=True, nn_idx=nb_idx_real, return_weight_only=True)  # b*h x n*k x 4
                nn_weights.append(nn_weight)
                idx_acc += querys[j].shape[1]

            nn_idxs = torch.stack(nn_idxs, dim=2).reshape(b*h, -1, k*l, 4)
            nn_weights = torch.stack(nn_weights, dim=2).reshape(b*h, -1, k*l, 4)
            attention_weight = attention_weights[i].permute(0, 2, 1, 4, 3).reshape(b*h, -1, k*l)
            sampled_values = MSDETRPCFunction.apply(nn_idxs, nn_weights, attention_weight, values).reshape(b, h, -1, c_).permute(0, 2, 1, 3).reshape(b, -1, c)
            outputs.append(sampled_values)

        outputs = [self.output_proj(output) for output in outputs]
        return outputs


class MSDeformAttnTransformerEncoderLayerPc(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 shepard_power=3.0, shepard_power_learnable=True):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttnPc(d_model, n_levels, n_heads, n_points, shepard_power, shepard_power_learnable)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        else:
            out = [t + p for t, p in zip(tensor, pos)]
            return out

    def forward_ffn(self, srcs):
        for i, src in enumerate(srcs):
            src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
            src = src + self.dropout3(src2)
            src = self.norm2(src)
            srcs[i] = src
        return srcs

    def forward(self, srcs, poss, spatial_shapes, pos_embeds, nb_idx):
        # self attention
        src2s = self.self_attn(self.with_pos_embed(srcs, pos_embeds), poss, srcs, spatial_shapes, nb_idx)
        for i, src2 in enumerate(src2s):
            src = srcs[i] + self.dropout1(src2)
            src2s[i] = self.norm1(src)
        srcs = src2s

        # ffn
        srcs = self.forward_ffn(srcs)

        return srcs


class MSDeformAttnTransformerEncoderPc(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, srcs, poss, spatial_shapes, pos_embeds, nb_idx):
        output = srcs
        for _, layer in enumerate(self.layers):
            output = layer(output, poss, spatial_shapes, pos_embeds, nb_idx)
        return output


class PointConv(nn.Module):
    def __init__(self, dim, out_dim, bias):
        super().__init__()
        inner_ch = 4
        self.weight_net = nn.Sequential(
            nn.Linear(5, inner_ch, bias=True),
            nn.LayerNorm(inner_ch),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(inner_ch*dim)
        self.linear = nn.Linear(dim*inner_ch, out_dim, bias=bias)


    def forward(self, inp):
        """
        Args:
            x - b x n x c, point cloud feature
            pos - b x n x 2, point cloud position
        Returns:
            feat - b x n x c_out, new feature
        """
        x, pos = inp
        b, n, c = x.shape
        nn_idx = knn_keops(pos, pos, 9)  # b x n x 9
        nn_pos = pos.gather(index=nn_idx.view(b, -1, 1).expand(-1, -1, 2), dim=1).reshape(b, n, 9, 2)
        rel_pos = pos.unsqueeze(2) - nn_pos

        global pre_table
        if not pre_table.is_cuda:
            pre_table = pre_table.to(rel_pos.device)
        weights_table = self.weight_net(pre_table)
        rel_shape = rel_pos.shape[:-1]
        rel_pos = (rel_pos.long().view(-1, 2) + rel_pos_width).clamp(0, table_width-1)
        pe_idx = rel_pos[..., 1]*table_width + rel_pos[..., 0]
        inner_ch = weights_table.shape[-1]
        weights = weights_table.gather(index=pe_idx.view(-1, 1).expand(-1, inner_ch), dim=0).reshape(*(rel_shape), inner_ch)

        feat = CLUSTENWFFunction.apply(weights, x, nn_idx).reshape(b, n, -1)  # b x n x ic*c

        feat = self.norm(feat)

        feat = self.linear(feat)  # b x n x c
        return feat


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        shepard_power: float,
        shepard_power_learnable: bool
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
            transformer_in_features: list of feature names into the deformable MSDETR
            common_stride: the stride of the finest feature map; outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
            shepard_power: the power used in deformable attn interpolation
            shepard_power_learnable: whether to make the power learnable
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]

        # this is the input shape of transformer encoder (could use less features than pixel decoder)
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Linear(in_channels, conv_dim, bias=True),
                    nn.LayerNorm(conv_dim)
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(transformer_in_channels[-1], conv_dim, bias=True),
                    nn.LayerNorm(conv_dim)
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnlyPc(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            shepard_power=shepard_power,
            shepard_power_learnable=shepard_power_learnable
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        self.mask_features = nn.Linear(
            conv_dim,
            mask_dim
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):

            lateral_conv = nn.Sequential(
                nn.Linear(in_channels, conv_dim, bias=True),
                nn.LayerNorm(conv_dim)
            )
            output_conv = nn.Sequential(
                PointConv(conv_dim, conv_dim, bias=True),
                nn.LayerNorm(conv_dim),
                nn.ReLU()
            )
            weight_init.c2_xavier_fill(lateral_conv[0])
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret['shepard_power'] = cfg.MODEL.AFF.SHEPARD_POWER / 2.0  # since the distances are already squared
        ret['shepard_power_learnable'] = cfg.MODEL.AFF.SHEPARD_POWER_LEARNABLE
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        """
        Args
            features - a dictionary of a list of point clouds with their features, positions and canvas sizes
        """
        srcs = []
        poss = []
        pos_embed = []
        spatial_shapes = []
        nb_idx = []
        finest_feat = self.in_features[0]
        grid_hw = features[finest_feat+"_spatial_shape"]
        hs = torch.arange(0, grid_hw[0], device=features[finest_feat].device)
        ws = torch.arange(0, grid_hw[1], device=features[finest_feat].device)
        ys, xs = torch.meshgrid(hs, ws)
        b = features[finest_feat].shape[0]
        grid_pos = torch.stack([xs, ys], dim=2).unsqueeze(0).expand(b, -1, -1, -1).reshape(b, -1, 2)  # b x h*w x 2
        # Reverse feature maps into top-down order (from low to high resolution) res5 to res3
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            pos = features[f+"_pos"].float()
            spatial_shape = features[f+"_spatial_shape"]
            srcs.append(self.input_proj[idx](x))
            poss.append(pos)
            pos_embed.append(self.pe_layer(pos))
            spatial_shapes.append(spatial_shape)
            scaled_pos = scale_pos(pos, spatial_shape, grid_hw, no_bias=True)
            nb_idx.append(knn_keops(grid_pos, scaled_pos, 4))
        last_pos = poss[-1]
        last_ss = spatial_shapes[-1]
        spatial_shapes.append(grid_hw)

        out = self.transformer(srcs, poss, spatial_shapes, pos_embed, nb_idx)

        multi_scale_features = []

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution) only res2
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            pos = features[f+"_pos"].float()
            spatial_shape = features[f+"_spatial_shape"]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            last_pos = scale_pos(last_pos, last_ss, spatial_shape, no_bias=True)
            y = cur_fpn + upsample_feature_shepard(pos, last_pos, out[-1], custom_kernel=True)
            y = output_conv((y, pos))
            last_pos = pos
            last_ss = spatial_shape
            out.append(y)

        num_cur_levels = 0
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), last_pos, out[0], multi_scale_features, poss[:self.maskformer_num_feature_levels]
