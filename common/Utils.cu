#include "Utils.h"

cudaError_t handleCudaErrors(cudaError_t error_code) {
    if(error_code != cudaSuccess) {
        std::cout << "CUDA-related error: " << cudaGetErrorString(error_code) << std::endl;
        exit(EXIT_FAILURE);
    }
    return error_code;
}


unsigned getNumberOfStreamingMultiprocessors() {
    int deviceId = 0;
    handleCudaErrors( cudaGetDevice(&deviceId) );
    cudaDeviceProp deviceProp{};
    handleCudaErrors( cudaGetDeviceProperties(&deviceProp, deviceId) );
    return static_cast<unsigned>(deviceProp.multiProcessorCount);
}


void printInformationAboutGPUDevice() {
    int deviceId = 0;
    handleCudaErrors( cudaGetDevice(&deviceId) );
    cudaDeviceProp deviceProp{};
    handleCudaErrors( cudaGetDeviceProperties(&deviceProp, deviceId) );
    std::cout << "[DEVICE] Name: " << deviceProp.name << ", number of SMs: " << deviceProp.multiProcessorCount
              << ", shared memory per SM: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
}