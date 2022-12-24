#include <torch/torch.h>

#include "voxel_pooling.h"

at::Tensor voxel_pooling_forward_cpu(const at::Tensor inputs, const at::Tensor idx,
                                     const at::Tensor counts) {
  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);
  at::Tensor out = torch::zeros(
      {N1, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  for (int i = 0; i < N; i++) {
    int pos = *(idx.data_ptr<int>() + i);
    if (pos < 0 || pos >= N1) continue;
#pragma omp parallel for
    for (int j = 0; j < c; j++) {
      *(out.data_ptr<float>() + pos * c + j) +=
          *(inputs.data_ptr<float>() + i * c + j) /
          (float)(*(counts.data_ptr<int>() + pos));
    }
  }
  return out;
}

at::Tensor voxel_pooling_backward_cpu(const at::Tensor top_grad,
                                      const at::Tensor idx, const at::Tensor counts,
                                      const int N) {
  int N1 = top_grad.size(0);
  int c = top_grad.size(1);
  at::Tensor bottom_grad = torch::zeros(
      {N, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  for (int i = 0; i < N; i++) {
    int pos = *(idx.data_ptr<int>() + i);
    if (pos < 0 || pos >= N1) continue;
#pragma omp parallel for
    for (int j = 0; j < c; j++) {
      *(bottom_grad.data_ptr<float>() + i * c + j) =
          *(top_grad.data_ptr<float>() + pos * c + j) /
          (float)(*(counts.data_ptr<int>() + pos));
    }
  }
  return bottom_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_pooling_forward_cpu", &voxel_pooling_forward_cpu);
  m.def("voxel_pooling_backward_cpu", &voxel_pooling_backward_cpu);
  m.def("voxel_pooling_forward_cuda", &voxel_pooling_forward_cuda);
  m.def("voxel_pooling_backward_cuda", &voxel_pooling_backward_cuda);
}
