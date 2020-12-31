#ifndef CUDA_DATA_MINING_CPP_STANDARDIZATIONCPU_H
#define CUDA_DATA_MINING_CPP_STANDARDIZATIONCPU_H

#include <vector>


/**
 * Performs data standardization on CPU by calculating the standard score (z-score) of data.
 * @param data_host Data stored in row-major order in RAM memory.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_rows Number of rows (entries) in the dataset.
 */
void performDataStandardizationOnCPU(std::vector<float>& data_host,
                                     unsigned number_of_columns, unsigned number_of_rows);


#endif //CUDA_DATA_MINING_CPP_STANDARDIZATIONCPU_H
