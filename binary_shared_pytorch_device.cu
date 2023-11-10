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

#define DSIZE sizeof(unsigned int)
#define DSIZE_BIT sizeof(unsigned int)*8 

#define TILE_SIZE 16

using namespace std;
using data_type = torch::Tensor; 

//TODO : Let us make it "Flexable"
//What Tile_size? should it be squared form?


__global__ void tile_encoding(float* mat1, float* mat2, float* mat3, int m, int n, int k){

    __shared__ unsigned int tile1[TILE_SIZE][TILE_SIZE]; //For 32 size, 32*32
    __shared__ unsigned int tile2[TILE_SIZE][TILE_SIZE]; //For 32 size, 32*32
    
    unsigned int row = threadIdx.y +  blockIdx.y * blockDim.y;
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    //To tile
    //unsigned int len = n/32;
    //printf("%d is tcount\n", tcount);
    unsigned int temp = 0;

    unsigned int tile1_start = n*row+ threadIdx.x;
    unsigned int tile2_start = threadIdx.y*k + column;
    for(int i = 0; i < n/TILE_SIZE; i++){
        //tile
        tile1[threadIdx.y][threadIdx.x] = mat1[tile1_start+i*TILE_SIZE];
        tile2[threadIdx.y][threadIdx.x] = mat2[tile2_start+i*TILE_SIZE*k];
        //tile1[threadIdx.y][threadIdx.x] = mat1[TILE_SIZE * blockIdx.y * n + threadIdx.y*n+threadIdx.x+i*TILE_SIZE];
        //tile2[threadIdx.y][threadIdx.x] = mat2[TILE_SIZE * blockIdx.x + threadIdx.y*k+threadIdx.x+i*TILE_SIZE*k]; 
        __syncthreads();
        //Calculation
        for(int j = 0; j < TILE_SIZE; j++){
            temp += __popc(tile1[threadIdx.y][j]^tile2[j][threadIdx.x]);
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
        mat3[row*k+column] = temp;
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

    float* d_mat1 = (float *)mat1.data<float>();
    float* d_mat2 = (float *)mat2.data<float>();
    float* d_mat3 = (float *)mat3.data<float>();


    dim3 block_main(16, 16); 
    dim3 grid_main(int(k/16), int(m/16));
    
    

    //Main Calculation here
    tile_encoding <<< grid_main, block_main>>> (d_mat1, d_mat2, d_mat3,m,n,k);
    cudaDeviceSynchronize();
    mat3 = torch::from_blob(d_mat3, {m*k}, torch::kInt32);
    cudaDeviceSynchronize();
}
