#ifndef CUDA_DATA_MINING_CPP_COMMONKERNELS_H
#define CUDA_DATA_MINING_CPP_COMMONKERNELS_H


/**
 * GPU kernel. Zero-initializes the device memory that contains floating-point numbers. This is done because
 * memset to 0 on floating points is not safe.
 * @param memory Pointer to GPU memory that should be initialized with zeros.
 * @param number_of_elements Number of elements to initialize.
 */
__global__ void zeroInitializeFloatMemory(float * memory, unsigned number_of_elements);


#endif //CUDA_DATA_MINING_CPP_COMMONKERNELS_H
