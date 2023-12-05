# binxor_test
__global__ void row_encoding_cu(float* inmatrix, float* outmatrix, int row){
    unsigned int temp = 0;
    unsigned int x = threadIdx.x +blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    for(int i = 0 ; i < 32; i ++){
        temp<<1;
        temp += inmatrix[x*32+i+y*row];
    }
    outmatrix[x+y*row] = temp;
}

torch::Tensor row_encoding(torch::Tensor mat1, int m, int n){
    dim3 block(1,32);
    dim3 grid((int((m+1)/32))-1, (int((n+1)/32))-1);
    row_encoding_cu <<< grid, block >>> (mat1, mat2, n);
}