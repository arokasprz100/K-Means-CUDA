#ifndef CUDA_DATA_MINING_CPP_STATISTICSGPU_H
#define CUDA_DATA_MINING_CPP_STATISTICSGPU_H


/**
 * Device function that computes the mean of data stored in shared memory using the reduction data access pattern.
 * @tparam BLOCK_SIZE Number of threads in a single block.
 * @param shared_mem_entries Pointer to shared memory where the mean will be computed.
 * @param thread Thread identifier (threadIdx.x).
 * @param number_of_entries Number of non-zero entries in the shared memory.
 * @param final_result Pointer to GPU memory where the final result should be stored.
 */
template <unsigned BLOCK_SIZE>
__device__ void computeMeanOfSharedMemoryEntries(float * shared_mem_entries, unsigned thread,
                                                 unsigned number_of_entries,
                                                 float * final_result) {

    // reduction engine - main reduction loop
    for(unsigned stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        __syncthreads();
        if(thread % stride == 0) {
            shared_mem_entries[2 * thread] += shared_mem_entries[2 * thread + stride];
        }
    }

    // accumulate the local results in the global one using atomic operations
    __syncthreads();
    if(thread == 0) {
        atomicAdd(final_result, (shared_mem_entries[0] / static_cast<float>(number_of_entries)));
    }

}


/**
 * Computes mean of given dataset column by performing the copy of column data to shared memory and using
 * the reduction pattern.
 * @tparam BLOCK_SIZE Number of threads in a single block.
 * @param data Pointer to GPU memory with data stored in row-major order.
 * @param input_size Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 * @param column Index of column whose mean should be computed.
 * @param final_result Pointer to GPU memory where the final result should be stored.
 */
template <unsigned BLOCK_SIZE>
__global__ void getMeanOfGivenColumn(const float * data, unsigned input_size, unsigned number_of_columns,
                                     unsigned column, float * final_result) {

    // shared memory to hold partial sums
    __shared__ float single_segment_partial_sums[2 * BLOCK_SIZE];

    // create helper variables
    unsigned thread = threadIdx.x;
    unsigned block = blockIdx.x;
    unsigned start = 2 * block * BLOCK_SIZE;

    // initialize the shared memory with input data (paying attention to boundary conditions)
    if(number_of_columns * thread + start + column < input_size * number_of_columns) {
        single_segment_partial_sums[thread] = data[number_of_columns * thread + start + column];
    } else {
        single_segment_partial_sums[thread] = 0.0f;
    }

    // same as above, but with the second element
    if(number_of_columns * (thread + BLOCK_SIZE) + start + column < input_size * number_of_columns) {
        single_segment_partial_sums[thread + BLOCK_SIZE] = data[number_of_columns * (thread + BLOCK_SIZE) + start + column];
    } else {
        single_segment_partial_sums[thread + BLOCK_SIZE] = 0.0f;
    }

    computeMeanOfSharedMemoryEntries<BLOCK_SIZE>(single_segment_partial_sums, thread, input_size, final_result);

}


/**
 * Computes variance of given dataset column by performing the copy of column data to shared memory and using the
 * reduction data access pattern.
 * @tparam BLOCK_SIZE Number of threads in a single block.
 * @param data Pointer to GPU memory with data stored in row-major order.
 * @param input_size Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 * @param column Index of column whose variance should be computed.
 * @param columns_means Pointer to GPU memory with means of each dataset column.
 * @param final_result Pointer to GPU memory where the final result should be stored.
 */
template <unsigned BLOCK_SIZE>
__global__ void getVarianceOfGivenColumn(const float * data, unsigned input_size, unsigned number_of_columns,
                                         unsigned column, float * columns_means, float * final_result) {

    // shared memory to hold partial sums
    __shared__ float single_segment_partial_sums[2 * BLOCK_SIZE];

    // create helper variables
    unsigned thread = threadIdx.x;
    unsigned block = blockIdx.x;
    unsigned start = 2 * block * BLOCK_SIZE;

    // initialize the shared memory with input data (paying attention to boundary conditions)
    if(number_of_columns * (thread + start) + column < input_size * number_of_columns) {
        single_segment_partial_sums[thread] = powf(columns_means[column] -
                data[number_of_columns * (thread + start) + column], 2.0);
    } else {
        single_segment_partial_sums[thread] = 0.0f;
    }

    // same as above, but with the second element
    if(number_of_columns * (thread + start + BLOCK_SIZE) + column < input_size * number_of_columns) {
        single_segment_partial_sums[thread + BLOCK_SIZE] = powf(columns_means[column] -
                data[number_of_columns * (thread + start + BLOCK_SIZE) + column], 2.0);
    } else {
        single_segment_partial_sums[thread + BLOCK_SIZE] = 0.0f;
    }

    computeMeanOfSharedMemoryEntries<BLOCK_SIZE>(single_segment_partial_sums, thread, input_size, final_result);
}




#endif //CUDA_DATA_MINING_CPP_STATISTICSGPU_H
