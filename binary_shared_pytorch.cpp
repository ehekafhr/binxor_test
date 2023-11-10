#include <torch/extension.h>
#include <iostream>
#include <vector>

void binary_xor_matmul_cu(torch::Tensor&, torch::Tensor&, torch::Tensor&);

void binxor(
    torch::Tensor& input1,
    torch::Tensor& input2,
    torch::Tensor& input3
){
    binary_xor_matmul_cu(input1, input2, input3);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("binxor", &binxor, "binxor");
}