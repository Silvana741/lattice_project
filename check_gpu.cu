#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
        return -1;
    }

    if (device_count == 0) {
        cout << "No CUDA-capable GPU detected!" << endl;
    } else {
        cout << "Number of CUDA-capable GPUs: " << device_count << endl;
    }

    return 0;
}
