#include <cuda_runtime.h>
#include <iostream>
#include <climits>
using namespace std;
#define THREADS_PER_BLOCK 1024

// Kernel to compute block-level minimums
__global__ void findBlockMinimum(const int *arr, int *block_min, int n) {
    __shared__ int shared_min[THREADS_PER_BLOCK];

    int tid = threadIdx.x;              // Thread index within the block
    int global_idx = blockIdx.x * blockDim.x + tid; // Global thread index

    // Load data from global memory into shared memory
    if (global_idx < n) {
        shared_min[tid] = arr[global_idx];
    } else {
        shared_min[tid] = INT_MAX; // If out of bounds, assign max value
    }
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
        }
        __syncthreads();
    }

    // The first thread in the block writes the block minimum to global memory
    if (tid == 0) {
        block_min[blockIdx.x] = shared_min[0];
    }
}

// Kernel to compute the global minimum from block minimums
__global__ void findGlobalMinimum(const int *block_min, int *global_min, int num_blocks) {
    __shared__ int shared_min[THREADS_PER_BLOCK];

    int tid = threadIdx.x;

    // Load block minimums into shared memory
    if (tid < num_blocks) {
        shared_min[tid] = block_min[tid];
    } else {
        shared_min[tid] = INT_MAX;
    }
    __syncthreads();

    // Perform reduction within the single block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
        }
        __syncthreads();
    }

    // The first thread writes the global minimum to global memory
    if (tid == 0) {
        *global_min = shared_min[0];
    }
}

int main() {
    const int n = 1 << 20; // 2^20 elements (1,048,576 elements)
    int *h_arr = new int[n];
    int *d_arr, *d_block_min, *d_global_min;

    // Initialize the array with random values
    srand(0);
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 1000000;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMalloc((void **)&d_block_min, THREADS_PER_BLOCK * sizeof(int)); // Block minimums
    cudaMalloc((void **)&d_global_min, sizeof(int)); // Global minimum

    // Copy array from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to compute block-level minimums
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    findBlockMinimum<<<num_blocks, THREADS_PER_BLOCK>>>(d_arr, d_block_min, n);

    // Launch kernel to compute the global minimum
    findGlobalMinimum<<<1, THREADS_PER_BLOCK>>>(d_block_min, d_global_min, num_blocks);

    // Copy the result back to host
    int h_global_min;
    cudaMemcpy(&h_global_min, d_global_min, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "The minimum value in the array is: " << h_global_min << std::endl;

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_block_min);
    cudaFree(d_global_min);

    // Free host memory
    delete[] h_arr;

    return 0;
}
