#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
from clusten import WEIGHTEDGATHERFunction

"""
Test the correctness of WeightedGather custom kernel
"""

b = 100
n = 50
n_ = 100
k = 4
c = 32

# dummy data
nn_idx = torch.randint(n_, (b, n, k)).cuda()
nn_weights = torch.rand(b, n, k).cuda()
feature = torch.rand(b, n_, c).cuda()
nn_weights.requires_grad_(True)
nn_weights.retain_grad()
feature.requires_grad_(True)
feature.retain_grad()

# use the custom kernel
up_features = WEIGHTEDGATHERFunction.apply(nn_idx, nn_weights, feature)
up_features.mean().backward()
grad_weights = nn_weights.grad.clone().detach()
grad_feat = feature.grad.clone().detach()
nn_weights.grad.data.zero_()
feature.grad.data.zero_()

# use the pytorch equivalent
nn_features = feature.gather(index=nn_idx.view(b, -1).unsqueeze(2).expand(-1, -1, c), dim=1).reshape(b, n, k, c)
up_features2 = nn_features.mul(nn_weights.unsqueeze(3).expand(-1, -1, -1, c)).sum(dim=2)  # b x n x c
up_features2.mean().backward()
grad_weights2 = nn_weights.grad.clone().detach()
grad_feat2 = feature.grad.clone().detach()
nn_weights.grad.data.zero_()
feature.grad.data.zero_()

print('diff of forward: ', torch.linalg.norm(up_features2 - up_features))
print('diff of grad weights: ', torch.linalg.norm(grad_weights2 - grad_weights))
print('diff of grad feat: ', torch.linalg.norm(grad_feat2 - grad_feat))
