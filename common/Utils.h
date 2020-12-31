#ifndef CUDA_DATA_MINING_CPP_UTILS_H
#define CUDA_DATA_MINING_CPP_UTILS_H

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>


/**
 * CUDA error handling function. If anything went wrong, prints the information and terminates the program.
 * @param error_code Result of CUDA API call.
 * @return Same code it has been given. Nothing, if program is terminated.
 */
cudaError_t handleCudaErrors(cudaError_t error_code);


/**
 * Performs device query to get the number of Streaming Multiprocessors on used GPU device.
 * @return Number of SMs on used GPU.
 */
unsigned getNumberOfStreamingMultiprocessors();


/**
 * Performs device query and prints basic information about used GPU device.
 */
void printInformationAboutGPUDevice();


/**
 * Computes elapsed time between two timestamps acquired using the chrono library. Result is in milliseconds.
 * @tparam T Type of timestamps.
 * @param start Timestamp of measurement start.
 * @param stop Timestamp of measurement stop.
 * @return Difference between both timestamps in milliseconds.
 */
template <typename T>
double computeElapsedTimeInMilliseconds(const T& start, const T& stop) {
    std::chrono::duration<float> duration = stop - start;
    std::chrono::nanoseconds elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    return elapsed_time.count() * 0.000001;
}


#endif //CUDA_DATA_MINING_CPP_UTILS_H
