#ifndef CUDA_DATA_MINING_CPP_STATISTICSCPU_H
#define CUDA_DATA_MINING_CPP_STATISTICSCPU_H

#include <vector>


/**
 * Computes means of dataset columns on a CPU.
 * @param data_host Data stored in row-major order in RAM memory.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @return Vector with means of each column.
 */
std::vector<float> computeColumnsMeansOnCPU(const std::vector<float>& data_host,
                                            unsigned number_of_columns, unsigned number_of_rows);


/**
 * Computes standard deviations of dataset columns on CPU.
 * @param data_host Data stored in row-major order in RAM memory.
 * @param columns_means Means of each dataset columns (computes using for example computeColumnsMeansOnCPU).
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @return Vector with standard deviations of each column.
 */
std::vector<float> computeColumnsStandardDeviationsOnCPU(const std::vector<float>& data_host,
                                                         const std::vector<float>& columns_means,
                                                         unsigned number_of_columns, unsigned number_of_rows);


#endif //CUDA_DATA_MINING_CPP_STATISTICSCPU_H
