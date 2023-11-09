#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor binary_xor_matmul_cu(torch::Tensor, torch::Tensor);

torch::Tensor binxor(
    torch::Tensor input1,
    torch::Tensor input2
){
    return binary_xor_matmul_cu(input1, input2);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("binxor", &binxor, "binxor");
}