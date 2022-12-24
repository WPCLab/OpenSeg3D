#include <torch/torch.h>

#include "knn_query.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_query_cuda", &knn_query_cuda);
}