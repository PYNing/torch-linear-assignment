#include <torch/extension.h>

std::vector<torch::Tensor> batch_linear_assignment_cuda(torch::Tensor cost);
std::vector<torch::Tensor> batch_linear_assignment_var_len_cuda_impl(std::vector<torch::Tensor>& costs);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Returns col4row and row4col.
std::vector<torch::Tensor> batch_linear_assignment(torch::Tensor cost) {
  CHECK_INPUT(cost);
  return batch_linear_assignment_cuda(cost);
}

std::vector<torch::Tensor> batch_linear_assignment_var_len(std::vector<torch::Tensor> costs) {
  for(const auto& cost:costs) {
      CHECK_INPUT(cost);
  }
  return batch_linear_assignment_var_len_cuda_impl(costs);
}

bool has_cuda() {
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("has_cuda", &has_cuda, "CUDA build flag.");
  m.def("batch_linear_assignment", &batch_linear_assignment, "Batch linear assignment (CUDA).");
  m.def("batch_linear_assignment_var_len", &batch_linear_assignment_var_len, "Batch linear assignment with vary length(CUDA).");
}
