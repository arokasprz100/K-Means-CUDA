#include "CommonKernels.h"

__global__ void zeroInitializeFloatMemory(float * memory, unsigned number_of_elements) {

    unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    for(unsigned i = start; i < number_of_elements; i += stride) {
        memory[i] = 0.0f;
    }
}
