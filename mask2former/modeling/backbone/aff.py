#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .point_utils import knn_keops, space_filling_cluster
from ..clusten import CLUSTENQKFunction, CLUSTENAVFunction, CLUSTENWFFunction

# assumes largest input resolution is 2048 x 2048
rel_pos_width = 2048 // 4 - 1
table_width = 2 * rel_pos_width + 1

pre_hs = torch.arange(table_width).float()-rel_pos_width
pre_ws = torch.arange(table_width).float()-rel_pos_width
pre_ys, pre_xs = torch.meshgrid(pre_hs, pre_ws)  # table_width x table_width

# expanded relative position lookup table
dis_table = (pre_ys**2 + pre_xs**2) ** 0.5
sin_table = pre_ys / dis_table
cos_table = pre_xs / dis_table
pre_table = torch.stack([pre_xs, pre_ys, dis_table, sin_table, cos_table], dim=2)  # table_width x table_width x 5
pre_table[torch.bitwise_or(pre_table.isnan(), pre_table.isinf()).nonzero(as_tuple=True)] = 0
pre_table = pre_table.reshape(-1, 5)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterAttention(nn.Module):
    """
    Performs local attention on nearest clusters

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        self.softmax = nn.Softmax(dim=-1)

        self.blank_k = nn.Parameter(torch.randn(dim))
        self.blank_v = nn.Parameter(torch.randn(dim))

        self.pos_embed = nn.Linear(self.pos_dim+3, num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        c_ = c // self.num_heads
        d = self.pos_dim
        assert c == self.dim, "dim does not accord to input"
        h = self.num_heads

        # get qkv
        q = self.q(feat)  # b x n x c
        q = q * self.scale
        kv = self.kv(feat)  # b x n x 2c

        # get attention
        if not global_attn:
            nbhd_size = member_idx.shape[-1]
            m = nbhd_size
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = CLUSTENQKFunction.apply(q, key, member_idx)  # b x h x n x m
            mask = cluster_mask
            if mask is not None:
                mask = mask.reshape(b, 1, n, m)
        else:
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)  # b x h x n x c_
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = q @ key.transpose(-1, -2)  # b x h x n x n
            mask = None

        # position embedding
        global pre_table
        if not pre_table.is_cuda:
            pre_table = pre_table.to(pe_idx.device)
        pe_table = self.pos_embed(pre_table)  # 111 x 111 x h

        pe_shape = pe_idx.shape
        pos_embed = pe_table.gather(index=pe_idx.view(-1, 1).expand(-1, h), dim=0).reshape(*(pe_shape), h).permute(0, 3, 1, 2)

        attn = attn + pos_embed

        if mask is not None:
            attn = attn + (1-mask)*(-100)

        # blank token
        blank_attn = (q * self.blank_k.reshape(1, h, 1, c_)).sum(-1, keepdim=True)  # b x h x n x 1
        attn = torch.cat([attn, blank_attn], dim=-1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        blank_attn = attn[..., -1:]
        attn = attn[..., :-1]
        blank_v = blank_attn * self.blank_v.reshape(1, h, 1, c_)  # b x h x n x c_

        # aggregate v
        if global_attn:
            feat = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)
        else:
            feat = CLUSTENAVFunction.apply(attn, v, member_idx).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)

        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        return feat

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class ClusterTransformerBlock(nn.Module):
    r""" Cluster Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0., layer_scale=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = ClusterAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        assert c == self.dim, "dim does not accord to input"

        shortcut = feat
        feat = self.norm1(feat)

        # cluster attention
        feat = self.attn(feat=feat,
                         member_idx=member_idx,
                         cluster_mask=cluster_mask,
                         pe_idx=pe_idx,
                         global_attn=global_attn)

        # FFN
        if not self.layer_scale:
            feat = shortcut + self.drop_path(feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(feat_mlp)
        else:
            feat = shortcut + self.drop_path(self.gamma1 * feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(self.gamma2 * feat_mlp)

        return feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"


class ClusterMerging(nn.Module):
    r""" Adaptive Downsampling.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, alpha=4.0, ds_rate=0.25, reserve_on=True):
        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.alpha = alpha
        self.ds_rate = ds_rate
        self.reserve_on = reserve_on

        # pointconv
        inner_ch = 4
        self.weight_net = nn.Sequential(
            nn.Linear(self.pos_dim+3, inner_ch, bias=True),
            nn.LayerNorm(inner_ch),
            nn.GELU()
        )

        self.norm = norm_layer(inner_ch*dim)
        self.linear = nn.Linear(dim*inner_ch, out_dim)

    def forward(self, pos, feat, member_idx, cluster_mask, learned_prob, stride, pe_idx, reserve_num):
        """
        Args:
            pos - b x n x 2, token positions
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            learned_prob - b x n x 1, learned importance scores
            stride - int, "stride" of the current feature map, 2,4,8 for the 3 stages respectively
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            reserve_num - int, number of tokens to be reserved
        """

        b, n, c = feat.shape
        d = pos.shape[2]

        keep_num = int(n*self.ds_rate)
        pos_long = pos.long()

        # grid prior
        if stride == 2:  # no ada ds yet, no need ada grid
            grid_prob = ((pos_long % stride) == 0).all(-1).float()  # b x n
        else:
            _, min_dist = knn_keops(pos, pos, 2, return_dist=True)  # b x n x 2
            min_dist = min_dist[:, :, 1]  # b x n
            ada_stride = 2**(min_dist.log2().ceil()+1)  # b x n
            grid_prob = ((pos_long % ada_stride.unsqueeze(2).long()) == 0).all(-1).float()  # b x n

        final_prob = grid_prob

        # add importance score
        if learned_prob is not None:
            lp = learned_prob.detach().view(b, n)
            lp = lp * self.alpha
            final_prob = final_prob + lp

        # reserve points on a coarse grid
        if self.reserve_on:
            reserve_mask = ((pos_long % (stride*2)) == 0).all(dim=-1).float()  # b x n
            final_prob = final_prob + (reserve_mask*(-100))
            sample_num = keep_num - reserve_num
        else:
            sample_num = keep_num

        sample_idx = final_prob.topk(sample_num, dim=1, sorted=False)[1]  # b x n_

        if self.reserve_on:
            reserve_idx = reserve_mask.nonzero(as_tuple=True)[1].reshape(b, reserve_num)
            idx = torch.cat([sample_idx, reserve_idx], dim=-1).unsqueeze(2)  # b x n_ x 1
        else:
            idx = sample_idx.unsqueeze(2)

        n = idx.shape[1]
        assert n == keep_num, "n not equal to keep num!"

        # gather pos, nbhd, nbhd position embedding, nbhd importance scores for topk merging locations
        pos = pos.gather(index=idx.expand(-1, -1, d), dim=1)  # b x n' x d

        nbhd_size = member_idx.shape[-1]
        member_idx = member_idx.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)  # b x n' x m
        pe_idx = pe_idx.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)  # b x n' x m
        if cluster_mask is not None:
            cluster_mask = cluster_mask.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)  # b x n' x m
        if learned_prob is not None:
            lp = learned_prob.gather(index=member_idx.view(b, -1, 1), dim=1).reshape(b, n, nbhd_size, 1)  # b x n x m x 1

        # pointconv weights
        global pre_table
        if not pre_table.is_cuda:
            pre_table = pre_table.to(pe_idx.device)
        weights_table = self.weight_net(pre_table)  # 111 x 111 x ic

        weight_shape = pe_idx.shape
        inner_ch = weights_table.shape[-1]
        weights = weights_table.gather(index=pe_idx.view(-1, 1).expand(-1, inner_ch), dim=0).reshape(*(weight_shape), inner_ch)

        if learned_prob is not None:
            if cluster_mask is not None:
                lp = lp * cluster_mask.unsqueeze(3)
            weights = weights * lp
        else:
            if cluster_mask is not None:
                weights = weights * cluster_mask.unsqueeze(3)

        # merging features
        feat = CLUSTENWFFunction.apply(weights, feat, member_idx.view(b, n, -1)).reshape(b, n, -1)  # b x n x ic*c
        feat = self.norm(feat)
        feat = self.linear(feat)  # b x n x 2c

        return pos, feat


class BasicLayer(nn.Module):
    """ AutoFocusFormer layer for one stage.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        cluster_size (int): Cluster size.
        nbhd_size (int): Neighbor size. If larger than or equal to number of tokens, perform global attention;
                            otherwise, rounded to the nearest multiples of cluster_size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        layer_scale (float, optional): Layer scale initial parameter. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, out_dim, cluster_size, nbhd_size,
                 depth, num_heads, mlp_ratio,
                 alpha=4.0, ds_rate=0.25, reserve_on=True,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=0.0, downsample=None):

        super().__init__()
        self.dim = dim
        self.nbhd_size = nbhd_size
        self.cluster_size = cluster_size
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            ClusterTransformerBlock(dim=dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    layer_scale=layer_scale,
                                    norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer, alpha=alpha, ds_rate=ds_rate, reserve_on=reserve_on)
        else:
            self.downsample = None

        # cache the clustering result for the first feature map since it is on grid
        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = None, None, None, None, None

        if downsample is not None:
            self.prob_net = nn.Linear(dim, 1)

    def forward(self, pos, feat, h, w, on_grid, stride):
        """
        Args:
            pos - b x n x 2, token positions
            feat - b x n x c, token features
            h,w - max height and width of token positions
            on_grid - bool, whether the tokens are still on grid; True for the first feature map
            stride - int, "stride" of the current token set; starts with 2, then doubles in each stage
        """
        b, n, d = pos.shape
        if not isinstance(b, int):
            b, n, d = b.item(), n.item(), d.item()  # make the flop analyzer happy
        c = feat.shape[2]
        assert self.cluster_size > 0, 'self.cluster_size must be positive'

        if self.nbhd_size >= n:
            global_attn = True
            member_idx, cluster_mask = None, None
        else:
            global_attn = False
            k = int(math.ceil(n / float(self.cluster_size)))  # number of clusters
            nnc = min(int(round(self.nbhd_size / float(self.cluster_size))), k)  # number of nearest clusters
            nbhd_size = self.cluster_size * nnc
            self.nbhd_size = nbhd_size  # if not global attention, then nbhd size is rounded to nearest multiples of cluster


        if global_attn:
            rel_pos = (pos[:, None, :, :]+rel_pos_width) - pos[:, :, None, :]  # b x n x n x d
        else:
            if k == n:
                cluster_mean_pos = pos
                member_idx = torch.arange(n, device=feat.device).long().reshape(1, n, 1).expand(b, -1, -1)  # b x n x 1
                cluster_mask = None
            else:
                if on_grid and self.training:
                    if self.cluster_mean_pos is None:
                        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = space_filling_cluster(pos, self.cluster_size, h, w, no_reorder=False)
                    pos, cluster_mean_pos, member_idx, cluster_mask = self.pos[:b], self.cluster_mean_pos[:b], self.member_idx[:b], self.cluster_mask
                    feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), self.reorder[:b].view(-1)].reshape(b, n, c)
                    if cluster_mask is not None:
                        cluster_mask = cluster_mask[:b]
                else:
                    pos, cluster_mean_pos, member_idx, cluster_mask, reorder = space_filling_cluster(pos, self.cluster_size, h, w, no_reorder=False)
                    feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), reorder.view(-1)].reshape(b, n, c)

            assert member_idx.shape[1] == k and member_idx.shape[2] == self.cluster_size, "member_idx shape incorrect!"

            nearest_cluster = knn_keops(pos, cluster_mean_pos, nnc)  # b x n x nnc

            m = self.cluster_size
            member_idx = member_idx.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n, nbhd_size)  # b x n x nnc*m
            if cluster_mask is not None:
                cluster_mask = cluster_mask.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n, nbhd_size)
            pos_ = pos.gather(index=member_idx.view(b, -1, 1).expand(-1, -1, d), dim=1).reshape(b, n, nbhd_size, d)
            rel_pos = pos_ - (pos.unsqueeze(2)-rel_pos_width)  # b x n x nbhd_size x d

        rel_pos = rel_pos.clamp(0, table_width-1)
        pe_idx = (rel_pos[..., 1] * table_width + rel_pos[..., 0]).long()

        for i_blk in range(len(self.blocks)):
            blk = self.blocks[i_blk]
            feat = blk(feat=feat,
                       member_idx=member_idx,
                       cluster_mask=cluster_mask,
                       pe_idx=pe_idx,
                       global_attn=global_attn)

        if self.downsample is not None:
            learned_prob = self.prob_net(feat).sigmoid()  # b x n x 1
            reserve_num = math.ceil(h/(stride*2)) * math.ceil(w/(stride*2))

            pos_down, feat_down = self.downsample(pos=pos, feat=feat,
                                                  member_idx=member_idx, cluster_mask=cluster_mask,
                                                  learned_prob=learned_prob, stride=stride,
                                                  pe_idx=pe_idx, reserve_num=reserve_num)

        if self.downsample is not None:
            return pos, feat, pos_down, feat_down
        else:
            return pos, feat, pos, feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 32.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        self.patch_size = 4

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(embed_dim//2)
        self.act1 = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x - b x c x h x w, input imgs
        """
        # padding
        ps = self.patch_size
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps))
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps))

        x = self.proj2(self.act1(self.bn(self.proj1(x))))

        b, c, h, w = x.shape
        if not isinstance(b, int):
            b, c, h, w = b.item(), c.item(), h.item(), w.item()
        x = x.flatten(2).transpose(1, 2)  # b x n x c
        if self.norm is not None:
            x = self.norm(x)

        hs = torch.arange(0, h, device=x.device)
        ws = torch.arange(0, w, device=x.device)
        ys, xs = torch.meshgrid(hs, ws)
        pos = torch.stack([xs, ys], dim=2).unsqueeze(0).expand(b, -1, -1, -1).reshape(b, -1, 2).to(x.dtype)

        return pos, x, h, w


class AFF(nn.Module):
    """

    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (tuple(int)): Feature dimension of each stage. Default: [32,128,256,512]
        cluster_size (int): Cluster size. Default: 8
        nbhd_size (tuple(int)): Neighborhood size of local attention of each stage. Default: [48,48,48,48]
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
        depths (tuple(int)): Depth of each AFF layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        layer_scale (float, optional): Layer scale initial parameter; turned off if 0.0. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        out_indices (tuple(int)): Indices of output feature maps.
    """

    def __init__(self, in_chans=3, embed_dim=[32, 128, 256, 512],
                 cluster_size=8, nbhd_size=[48, 48, 48, 48],
                 alpha=4.0, ds_rate=0.25, reserve_on=True,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=2., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 layer_scale=0.0,
                 downsample=ClusterMerging,
                 out_indices=(0, 1, 2, 3)
                 ):

        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices

        num_features = embed_dim
        self.num_features = num_features

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim[i_layer]),
                               out_dim=int(embed_dim[i_layer+1]) if (i_layer < self.num_layers - 1) else None,
                               cluster_size=cluster_size,
                               nbhd_size=nbhd_size[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               alpha=alpha,
                               ds_rate=ds_rate,
                               reserve_on=reserve_on,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=downsample if (i_layer < self.num_layers - 1) else None,
                               layer_scale=layer_scale,
                               )
            self.layers.append(layer)


        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)


    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        '''
        x - b x c x h x w
        '''
        pos, x, h, w = self.patch_embed(x)  # b x n x c, b x n x d
        x = self.pos_drop(x)
        spatial_shape = (h, w)

        outs = {}
        for i_layer in range(self.num_layers):
            layer = self.layers[i_layer]
            pos_out, x_out, pos, x = layer(pos, x, h=h, w=w, on_grid=i_layer == 0, stride=2**(i_layer+1))

            if i_layer in self.out_indices:
                norm_layer = getattr(self, f"norm{i_layer}")
                x_out = norm_layer(x_out)

                outs["res{}".format(i_layer + 2)] = x_out
                outs["res{}_pos".format(i_layer + 2)] = pos_out
                outs["res{}_spatial_shape".format(i_layer + 2)] = spatial_shape

        return outs


@BACKBONE_REGISTRY.register()
class AutoFocusFormer(AFF, Backbone):
    def __init__(self, cfg, input_shape):

        in_chans = 3
        embed_dim = cfg.MODEL.AFF.EMBED_DIM
        depths = cfg.MODEL.AFF.DEPTHS
        num_heads = cfg.MODEL.AFF.NUM_HEADS
        mlp_ratio = cfg.MODEL.AFF.MLP_RATIO
        drop_rate = cfg.MODEL.AFF.DROP_RATE
        attn_drop_rate = cfg.MODEL.AFF.ATTN_DROP_RATE
        drop_path_rate = cfg.MODEL.AFF.DROP_PATH_RATE
        norm_layer = nn.LayerNorm
        patch_norm = cfg.MODEL.AFF.PATCH_NORM

        cluster_size = cfg.MODEL.AFF.CLUSTER_SIZE
        nbhd_size = cfg.MODEL.AFF.NBHD_SIZE
        layer_scale = cfg.MODEL.AFF.LAYER_SCALE
        alpha = cfg.MODEL.AFF.ALPHA
        ds_rate = cfg.MODEL.AFF.DS_RATE
        reserve_on = cfg.MODEL.AFF.RESERVE

        super().__init__(
            in_chans=in_chans,
            embed_dim=embed_dim,
            cluster_size=cluster_size,
            nbhd_size=nbhd_size,
            alpha=alpha,
            ds_rate=ds_rate,
            reserve_on=reserve_on,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            layer_scale=layer_scale,
        )

        self._out_features = cfg.MODEL.AFF.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B,C,H,W)
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"AFF takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        y = super().forward(x)
        return y

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
