/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>
#include <vector>

torch::Tensor msdetrpc_cuda_forward(
    const torch::Tensor &nn_idx,                            // b x n x m x k
    const torch::Tensor &nn_weight,                         // b x n x m x k
    const torch::Tensor &attn,                              // b x n x m
    const torch::Tensor &val);                              // b x n_ x c

std::vector<torch::Tensor> msdetrpc_cuda_backward(
    const torch::Tensor &d_feat, 
    const torch::Tensor &nn_idx,
    const torch::Tensor &nn_weight,
    const torch::Tensor &attn,
    const torch::Tensor &val);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor msdetrpc_forward(           
    const torch::Tensor &nn_idx,
    const torch::Tensor &nn_weight,
    const torch::Tensor &attn,
    const torch::Tensor &val) {
    CHECK_INPUT(nn_idx);
    CHECK_INPUT(nn_weight);
    CHECK_INPUT(attn);
    CHECK_INPUT(val);
    return msdetrpc_cuda_forward(nn_idx, nn_weight, attn, val);
}

std::vector<torch::Tensor> msdetrpc_backward(
    const torch::Tensor &d_feat,
    const torch::Tensor &nn_idx,
    const torch::Tensor &nn_weight,
    const torch::Tensor &attn,
    const torch::Tensor &val) {
    CHECK_INPUT(d_feat);
    CHECK_INPUT(nn_idx);
    CHECK_INPUT(nn_weight);
    CHECK_INPUT(attn);
    CHECK_INPUT(val);
    return msdetrpc_cuda_backward(d_feat, nn_idx, nn_weight, attn, val);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &msdetrpc_forward, "MSDETRPC forward (CUDA)");
  m.def("backward", &msdetrpc_backward, "MSDETRPC backward (CUDA)");
}
