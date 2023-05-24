#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
from clusten import MSDETRPCFunction

"""
Test the correctness of MSDETR (point cloud) custom kernel
"""

b = 100
n = 50
n_ = 100
m = 8
k = 4
c = 32

# dummy data
nn_idx = torch.randint(n_, (b, n, m, k)).cuda()
nn_weights = torch.rand(b, n, m, k).cuda()
attn = torch.rand(b, n, m).cuda()
val = torch.rand(b, n_, c).cuda()

nn_weights.requires_grad_(True)
nn_weights.retain_grad()
attn.requires_grad_(True)
attn.retain_grad()
val.requires_grad_(True)
val.retain_grad()

# use the custom kernel
feat = MSDETRPCFunction.apply(nn_idx, nn_weights, attn, val)
feat.mean().backward()
grad_weights = nn_weights.grad.clone().detach()
grad_attn = attn.grad.clone().detach()
grad_val = val.grad.clone().detach()
nn_weights.grad.data.zero_()
attn.grad.data.zero_()
val.grad.data.zero_()

# use the pytorch equivalent
nn_val = val.gather(index=nn_idx.view(b, -1).unsqueeze(2).expand(-1, -1, c), dim=1).reshape(b, n, m, k, c)
feat2 = ((nn_val * nn_weights.unsqueeze(4)).sum(3) * attn.unsqueeze(3)).sum(2)  # b x n x c
feat2.mean().backward()
grad_weights2 = nn_weights.grad.clone().detach()
grad_attn2 = attn.grad.clone().detach()
grad_val2 = val.grad.clone().detach()
nn_weights.grad.data.zero_()
attn.grad.data.zero_()
val.grad.data.zero_()

print('diff of forward: ', torch.linalg.norm(feat2 - feat))
print('diff of grad weights: ', torch.linalg.norm(grad_weights2 - grad_weights))
print('diff of grad attn: ', torch.linalg.norm(grad_attn2 - grad_attn))
print('diff of grad val: ', torch.linalg.norm(grad_val2 - grad_val))
