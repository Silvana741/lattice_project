#include <cuda_runtime.h>
#include <iostream>
#include <climits>

__device__ int global_min; // Global minimum variable

__global__ void findGlobalMinimum(const int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (idx >= n) return; // Ensure thread does not go out of bounds

    // Initialize global_min in the first thread (blockIdx.x == 0 && threadIdx.x == 0)
    if (idx == 0) {
        global_min = INT_MAX;
    }
    __syncthreads();

    // Each thread tries to update the global minimum
    int local_value = arr[idx];
    atomicMin(&global_min, local_value);

    // Wait until all threads have updated the global minimum
    __syncthreads();
}

int main() {
    // Example array
    const int n = 8;
    int h_arr[n] = {7, 3, 1, 8, 4, 2, 5, 6};
    int *d_arr;

    // Allocate memory on the device
    cudaMalloc((void **)&d_arr, n * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    findGlobalMinimum<<<blocks, threads_per_block>>>(d_arr, n);

    // Copy the global minimum from device to host
    int h_global_min;
    cudaMemcpyFromSymbol(&h_global_min, global_min, sizeof(int), 0, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "The minimum value in the array is: " << h_global_min << std::endl;

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
