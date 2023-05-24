/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>
#include <vector>

torch::Tensor weighted_gather_cuda_forward(
    const torch::Tensor &nbhd_idx,                          // b x n x m
    const torch::Tensor &weights,                           // b x n x m
    const torch::Tensor &feat);                             // b x n_ x c

std::vector<torch::Tensor> weighted_gather_cuda_backward(
    const torch::Tensor &d_feat_new, 
    const torch::Tensor &nbhd_idx,
    const torch::Tensor &weights,
    const torch::Tensor &feat);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor weighted_gather_forward(           
    const torch::Tensor &nbhd_idx,
    const torch::Tensor &weights,
    const torch::Tensor &feat) {
    CHECK_INPUT(nbhd_idx);
    CHECK_INPUT(weights);
    CHECK_INPUT(feat);
    return weighted_gather_cuda_forward(nbhd_idx, weights, feat);
}

std::vector<torch::Tensor> weighted_gather_backward(
    const torch::Tensor &d_feat_new,
    const torch::Tensor &nbhd_idx,
    const torch::Tensor &weights,
    const torch::Tensor &feat) {
    CHECK_INPUT(d_feat_new);
    CHECK_INPUT(nbhd_idx);
    CHECK_INPUT(weights);
    CHECK_INPUT(feat);
    return weighted_gather_cuda_backward(d_feat_new, nbhd_idx, weights, feat);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &weighted_gather_forward, "WEIGHTEDGATHER forward (CUDA)");
  m.def("backward", &weighted_gather_backward, "WEIGHTEDGATHER backward (CUDA)");
}
