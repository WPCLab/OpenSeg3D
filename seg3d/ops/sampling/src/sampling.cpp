#include <torch/torch.h>

#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthestsampling_cuda", &furthestsampling_cuda);
}
