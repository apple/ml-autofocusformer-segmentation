/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

#define CUDA_NUM_THREADS 1024

template <typename scalar_t>
__global__ void msdetrpc_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<int64_t,4,torch::DefaultPtrTraits> nn_idx,              // b x n x m x k
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> nn_weight,          // b x n x m x k
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> attn,               // b x n x m
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> val,                // b x n_ x c
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,                     // b x n x c
    const int length,               // n
    const int length_val,           // n_
    const int batch_size,           // b
    const int nbhd_size,            // m
    const int interp_size,          // k
    const int dim) {                // c

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                int64_t nbi;
                scalar_t updt = scalar_t(0);
                scalar_t updt_interp = scalar_t(0);
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    updt_interp = scalar_t(0);
                    for (unsigned int ki=0; ki < interp_size; ++ki) {
                        nbi = nn_idx[b][i][ni][ki];
                        updt_interp += val[b][nbi][c] * nn_weight[b][i][ni][ki];
                    }
                    updt += attn[b][i][ni] * updt_interp;
                }
                feat[b][i][c] = updt;
            }
        }
    }
}


torch::Tensor msdetrpc_cuda_forward(
    const torch::Tensor &nn_idx,             
    const torch::Tensor &nn_weight, 
    const torch::Tensor &attn,             
    const torch::Tensor &val) { 

    int64_t batch_size = nn_idx.size(0);
    int64_t length = nn_idx.size(1);
    int64_t nbhd_size = nn_idx.size(2);
    int64_t interp_size = nn_idx.size(3);
    int64_t length_val = val.size(1);
    int64_t dim = val.size(2);

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    auto feat = torch::zeros(
            {batch_size, length, dim}, val.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (batch_size + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn.scalar_type(), "msdetrpc_cuda_forward", ([&] {
        const auto nn_idx_a = nn_idx.packed_accessor32<int64_t,4,torch::DefaultPtrTraits>();
        const auto nn_weight_a = nn_weight.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto val_a = val.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();

        msdetrpc_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                nn_idx_a, nn_weight_a, attn_a, val_a, feat_a,
                length, length_val, batch_size, nbhd_size, interp_size, dim);
    }));
    return feat;
}


template <typename scalar_t>
__global__ void msdetrpc_val_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_feat,
    const torch::PackedTensorAccessor32<int64_t,4,torch::DefaultPtrTraits> nn_idx,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> nn_weight,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> attn,
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_val,
    const int length,
    const int length_val,
    const int batch_size,
    const int nbhd_size,
    const int interp_size,
    const int dim,
    const size_t d_val_numel) {

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                int64_t nbi;
                size_t index;
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    for (unsigned int ki=0; ki < interp_size; ++ki) {
                        nbi = nn_idx[b][i][ni][ki];

                        // d_val = d_feat * weight * attn
                        index = b*d_val.stride(0) + nbi*d_val.stride(1) + c;
                        at::native::fastAtomicAdd(d_val.data(), index, d_val_numel, d_feat[b][i][c] * nn_weight[b][i][ni][ki] * attn[b][i][ni], true);
                        // atomicAdd(&(d_val[b][nbi][c]), d_feat[b][i][c] * nn_weight[b][i][ni][ki] * attn[b][i][ni]); // avoid race condition
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void msdetrpc_attn_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_feat,
    const torch::PackedTensorAccessor32<int64_t,4,torch::DefaultPtrTraits> nn_idx,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> nn_weight,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> val,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_nn_weight,
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_attn,
    const int length,
    const int length_val,
    const int batch_size,
    const int nbhd_size,
    const int interp_size,
    const int dim) {

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int ni = blockIdx.x * blockDim.x + threadIdx.x;
            if (ni < nbhd_size){
                scalar_t updt_attn = scalar_t(0);
                scalar_t updt_weight;
                int64_t nbi;
                scalar_t tmp, weight_tmp;
                scalar_t attn_tmp = attn[b][i][ni];
                for (unsigned int ki=0; ki < interp_size; ++ki) {
                    updt_weight = scalar_t(0);
                    nbi = nn_idx[b][i][ni][ki];
                    weight_tmp = nn_weight[b][i][ni][ki];
                    #pragma unroll
                    for (unsigned int c=0; c < dim; ++c) {
                        tmp = val[b][nbi][c] * d_feat[b][i][c];
                        updt_weight += tmp * attn_tmp;
                        updt_attn += tmp * weight_tmp;
                    }
                    d_nn_weight[b][i][ni][ki] = updt_weight;
                }
                d_attn[b][i][ni] = updt_attn;
            }
        }
    }
}

std::vector<torch::Tensor> msdetrpc_cuda_backward(
    const torch::Tensor &d_feat,
    const torch::Tensor &nn_idx,
    const torch::Tensor &nn_weight,
    const torch::Tensor &attn,
    const torch::Tensor &val) {

    int64_t batch_size = nn_idx.size(0);
    int64_t length = nn_idx.size(1);
    int64_t nbhd_size = nn_idx.size(2);
    int64_t interp_size = nn_idx.size(3);
    int64_t length_val = val.size(1);
    int64_t dim = val.size(2);

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS* CHANNELTHREADS));

    int NBHDTHREADS = min(int64_t(CUDA_NUM_THREADS), nbhd_size);
    int TOKENTHREADS_NB = min(int64_t(CUDA_NUM_THREADS / NBHDTHREADS), length);
    int BATCHTHREADS_NB = max(1, CUDA_NUM_THREADS / (TOKENTHREADS_NB* NBHDTHREADS));

    auto d_nn_weight = torch::zeros_like(nn_weight);
    auto d_attn = torch::zeros_like(attn);
    auto d_val = torch::zeros_like(val);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (batch_size + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    const dim3 blocks_nb(
            (nbhd_size + NBHDTHREADS - 1) / NBHDTHREADS,
            (length + TOKENTHREADS_NB - 1) / TOKENTHREADS_NB,
            (batch_size + BATCHTHREADS_NB - 1) / BATCHTHREADS_NB);
    const dim3 threads_nb(NBHDTHREADS, TOKENTHREADS_NB, BATCHTHREADS_NB);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn.scalar_type(), "msdetrpc_cuda_backward", ([&] {
        const auto d_feat_a = d_feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto nn_idx_a = nn_idx.packed_accessor32<int64_t,4,torch::DefaultPtrTraits>();
        const auto nn_weight_a = nn_weight.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto val_a = val.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto d_nn_weight_a = d_nn_weight.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto d_val_a = d_val.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();

        const size_t d_val_numel = d_val.numel();
        msdetrpc_val_cuda_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                d_feat_a, nn_idx_a, nn_weight_a, attn_a, d_val_a,
                length, length_val, batch_size, nbhd_size, interp_size, dim, d_val_numel);
        msdetrpc_attn_cuda_backward_kernel<scalar_t><<<blocks_nb, threads_nb, 0, stream>>>(
                d_feat_a, nn_idx_a, nn_weight_a, attn_a, val_a, d_nn_weight_a,  d_attn_a,
                length, length_val, batch_size, nbhd_size, interp_size, dim);
    }));

    return {d_nn_weight, d_attn, d_val};
}
