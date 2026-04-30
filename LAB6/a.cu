#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA compatible devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("Compute capability:          %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Maximum block dimensions:    (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
                                                              deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum grid dimensions:     (%d, %d, %d)\n", deviceProp.maxGridSize[0],
                                                              deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum threads per block:   %d\n", deviceProp.maxThreadsPerBlock);
        printf("Total global memory:         %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Shared memory per block:     %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total constant memory:       %lu bytes\n", deviceProp.totalConstMem);
        printf("Warp size:                   %d\n", deviceProp.warpSize);
        printf("Concurrent kernels:          %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
    }
    return 0;
}
