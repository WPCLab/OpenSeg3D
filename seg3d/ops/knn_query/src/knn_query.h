#ifndef _KNN_QUERY
#define _KNN_QUERY

#include <vector>

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void knn_query_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void knn_query_cuda_launcher(int m, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2);

#ifdef __cplusplus
}
#endif

#endif