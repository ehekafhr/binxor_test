#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <cuda.h>

#include <torch/extension.h>

#define TILE_SIZE 16

using namespace std;
using data_type = torch::Tensor; 


__global__ void zip_column(int* mat1, int* mat2){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int basic_idx = idx*32;
    int retval = 0;
    for(int i = 0; i<32; i++){
        retval = retval | ((mat1[basic_idx+i])<<(31-i));
    }
    mat2[idx] = retval;
}

__global__ void zip_row(int* mat1, int* mat2, int row){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int basic_idx = x+y*row*32;
    int retval = 0;
    for(int i = 0; i<32; i++){
        retval = retval | (mat1[basic_idx+i*row]<<(31-i));
    }
    mat2[x+y*row] = retval;
}

torch::Tensor __zip_column_cu(torch::Tensor& mat1){
    unsigned int m = mat1.size(0);
    unsigned int n = mat1.size(1);
    torch::Tensor result = torch::zeros({m,1+(n-1)/32}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    int* data = (int*)mat1.data<int>();
    int* zdata = (int*)result.data<int>();
    zip_column <<< 1+(m*n-1)/512,16  >>> (data, zdata);
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor __zip_row_cu(torch::Tensor& mat1, int row){
    unsigned int m = mat1.size(0);
    unsigned int n = mat1.size(1);
    torch::Tensor result = torch::zeros({1+(m-1)/32,n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    int* data = (int*)mat1.data<int>();
    int* zdata = (int*)result.data<int>();
    zip_row <<<  (m/32,n/32),(32,1) >>> (data, zdata, row);
    cudaDeviceSynchronize();
    return result;
}


__global__ void tile_encoding(int* mat1, int* mat2, int* mat3, int m, int n, int k){

    __shared__ unsigned int tile1[TILE_SIZE][TILE_SIZE]; //For 32 size, 32*32
    __shared__ unsigned int tile2[TILE_SIZE][TILE_SIZE]; //For 32 size, 32*32
    
    unsigned int row = threadIdx.y +  blockIdx.y * blockDim.y;
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    //To tile
    //unsigned int len = n/32;
    //printf("%d is tcount\n", tcount);
    unsigned int temp = 0;

    unsigned int tile1_start = n*threadIdx.y+ column;
    unsigned int tile2_start = row*k + threadIdx.x;
    for(int i = 0; i < n/TILE_SIZE; i++){
        //tile
        if((threadIdx.x+i*TILE_SIZE)<n && row<m){
            tile1[threadIdx.y][threadIdx.x] = mat1[tile1_start+i*TILE_SIZE];
        }
        else{
            tile1[threadIdx.y][threadIdx.x] = 0;
        }
        if(column<k && (i*TILE_SIZE+threadIdx.y)<n){
            tile2[threadIdx.y][threadIdx.x] = mat2[tile2_start+i*TILE_SIZE*k];
        }
        else{
            tile2[threadIdx.y][threadIdx.x] = 0;
        }
       
        //tile1[threadIdx.y][threadIdx.x] = mat1[TILE_SIZE * blockIdx.y * n + threadIdx.y*n+threadIdx.x+i*TILE_SIZE];
        //tile2[threadIdx.y][threadIdx.x] = mat2[TILE_SIZE * blockIdx.x + threadIdx.y*k+threadIdx.x+i*TILE_SIZE*k]; 
        __syncthreads();
        //Calculation
        for(int j = 0; j < TILE_SIZE; j++){
            temp += __popc(tile1[threadIdx.y][j]^tile2[j][threadIdx.x]); // 다 0이되는 문제 발생.
        }
        //temp += __popc(tile1[threadIdx.y]^tile2[threadIdx.x]);
        __syncthreads();
    }
    //SHOULD THINK PITCH!
    //Edge Case..? can we handle it?
    if(temp>n*16){
        mat3[row*k+column] = 1;
    }
    else{
        mat3[row*k+column] = 0;
    }
}

//input is float matrix ( mat1), float matrix(mat2) (should be transposed), float matrix mat3(output)
//Size: mat1: (32*m) * n, mat2: n * (32*k). result matrix size: (32*m)*(32*k)
//m = 32x. n = x. k = 32x.
void binary_xor_matmul_cu(torch::Tensor& mat1, torch::Tensor& mat2, torch::Tensor& mat3){
    //First, Mat1/2/3 to global Mem.

    unsigned int m = mat1.size(0);
    unsigned int n = mat1.size(1);
    unsigned int k = mat2.size(1); 

    //torch::Tensor mat3 = torch::zeros({m*k}, torch::dtype(torch::kInt32).device(torch::kCPU));

    torch::Tensor z_mat1 = __zip_column_cu(mat1);
    torch::Tensor z_mat2 = __zip_row_cu(mat2, k);
    cudaDeviceSynchronize();

    int* z_int1 =(int*)z_mat1.data<int>();
    int* z_int2 =(int*)z_mat2.data<int>();
    int* d_mat3 = (int*)mat3.data<int>();

    dim3 block_main(16, 16); 
    dim3 grid_main(int(1+(z_mat2.size(1)-1)/16), int(1+(z_mat1.size(1)-1)/16));

    //Main Calculation here
    
    tile_encoding <<< grid_main, block_main>>> (z_int1, z_int2, d_mat3,z_mat1.size(0),z_mat1.size(1),z_mat2.size(1));
    cudaDeviceSynchronize();
    mat3 = torch::from_blob(d_mat3, {m*k}, torch::kInt32);
    cudaDeviceSynchronize();
}
