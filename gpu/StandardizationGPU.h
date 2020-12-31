#ifndef CUDA_DATA_MINING_CPP_STANDARDIZATIONGPU_H
#define CUDA_DATA_MINING_CPP_STANDARDIZATIONGPU_H


/**
 * Computes columns statistics and performs data standardization on GPU by calculating the standard score.
 * @param data_device Pointer to data in GPU memory that stores entries in row-major order.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_rows Number of rows (entries) in the dataset.
 */
void performDataStandardizationOnGPU(float * data_device, unsigned number_of_columns, unsigned number_of_rows);


/**
 * GPU kernel. Performs data standardization by calculating the standard score, based on given means and variances
 * of each dataset column.
 * @param data Pointer to data in GPU memory that stores entries in row-major order.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @param means Pointer to GPU memory with means of each dataset column.
 * @param variances Pointer to GPU memory with variances of each dataset column.
 */
__global__
void standardizeData(float * data, unsigned number_of_columns, unsigned number_of_rows,
                     const float * means, const float * variances);


#endif //CUDA_DATA_MINING_CPP_STANDARDIZATIONGPU_H
