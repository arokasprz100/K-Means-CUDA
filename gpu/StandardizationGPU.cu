#include "StandardizationGPU.h"

#include "StatisticsGPU.h"
#include "CommonKernels.h"
#include "../common/Utils.h"

#include <iostream>



void performDataStandardizationOnGPU(float * data_device, unsigned number_of_columns, unsigned number_of_rows) {

    // used block size
    constexpr unsigned block_size = 256;

    // perform device query - get number of SMs, execution configuration
    unsigned number_of_multiprocessors = getNumberOfStreamingMultiprocessors();
    unsigned number_of_blocks = number_of_multiprocessors * 2;
    std::cout << "[DEVICE] Execution configuration: (blocks: " << number_of_blocks << ", threads: " << block_size << ")" << std::endl;


    // adjust number of blocks used by reduction algorithms
    auto number_of_blocks_reduction = static_cast<unsigned>(number_of_rows / (block_size * 2));
    if(number_of_rows % (block_size * 2) != 0) {
        number_of_blocks_reduction += 1;
    }
    std::cout << "[DEVICE] Execution configuration for reductions: (blocks: " << number_of_blocks_reduction << ", threads: " << block_size << ")" << std::endl;


    // create time-measurement events
    cudaEvent_t time_start, time_stop;
    handleCudaErrors( cudaEventCreate(&time_start) );
    handleCudaErrors( cudaEventCreate(&time_stop) );

    // record the starting event
    handleCudaErrors( cudaEventRecord(time_start) );


    // allocate memory for means of each column and set to zeros
    float * columns_means_device = nullptr;
    handleCudaErrors( cudaMalloc(&columns_means_device, number_of_columns * sizeof(float)) );
    zeroInitializeFloatMemory<<<1, 1>>>(columns_means_device, number_of_columns);
    handleCudaErrors( cudaGetLastError() );
    handleCudaErrors( cudaDeviceSynchronize() );


    // allocate memory for standard deviations of each column and set to zeros
    float * columns_variances_device = nullptr;
    handleCudaErrors( cudaMalloc(&columns_variances_device, number_of_columns * sizeof(float)) );
    zeroInitializeFloatMemory<<<1, 1>>>(columns_variances_device, number_of_columns);
    handleCudaErrors( cudaGetLastError() );
    handleCudaErrors( cudaDeviceSynchronize() );


    // compute means for each column on GPU
    for(unsigned column = 0; column < number_of_columns; ++column) {
        getMeanOfGivenColumn<block_size><<<number_of_blocks_reduction, block_size>>>(data_device, number_of_rows, number_of_columns,
                                                                                     column, &columns_means_device[column]);
        handleCudaErrors( cudaGetLastError() );
    }
    handleCudaErrors( cudaDeviceSynchronize() );


    // compute standard deviations for each column on GPU
    for(unsigned column = 0; column < number_of_columns; ++column) {
        getVarianceOfGivenColumn<block_size><<<number_of_blocks_reduction, block_size>>>(data_device, number_of_rows, number_of_columns,
                                                                                         column, columns_means_device, &columns_variances_device[column]);
        handleCudaErrors( cudaGetLastError() );
    }
    handleCudaErrors( cudaDeviceSynchronize() );


    // standardize data on GPU
    standardizeData<<<number_of_blocks, block_size>>>(data_device, number_of_columns, number_of_rows, columns_means_device, columns_variances_device);
    handleCudaErrors( cudaGetLastError() );
    handleCudaErrors( cudaDeviceSynchronize() );


    // record the stopping event and synchronize
    handleCudaErrors( cudaEventRecord(time_stop) );
    handleCudaErrors( cudaEventSynchronize(time_stop) );

    // compute and display elapsed time
    float elapsed_time = 0.0f;
    handleCudaErrors( cudaEventElapsedTime(&elapsed_time, time_start, time_stop) );
    std::cout << "[DEVICE] Standardization - total elapsed time according to CUDA events: " << elapsed_time << " ms" << std::endl;


    // destroy the event objects
    handleCudaErrors( cudaEventDestroy(time_start) );
    handleCudaErrors( cudaEventDestroy(time_stop) );

    // free allocated memory
    handleCudaErrors( cudaFree(columns_means_device) );
    handleCudaErrors( cudaFree(columns_variances_device) );
}


__global__ void standardizeData(float * data, unsigned number_of_columns, unsigned number_of_rows,
                                const float * means, const float * variances) {

    unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    for(unsigned i = start; i < number_of_rows; i += stride) {
        unsigned data_index = i * number_of_columns;
        for(unsigned column = 0; column < number_of_columns; ++column) {
            data[data_index + column] = (data[data_index + column] - means[column]) / sqrt(variances[column]);
        }
    }
}