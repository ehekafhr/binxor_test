#include <torch/extension.h>

torch::Tensor binary_xor_matmul_cu(torch::Tensor mat1, torch::Tensor mat2);