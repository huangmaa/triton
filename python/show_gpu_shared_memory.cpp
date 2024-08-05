#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;

    return 0;
}
