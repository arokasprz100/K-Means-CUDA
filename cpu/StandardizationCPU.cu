#include "StandardizationCPU.h"
#include "StatisticsCPU.h"
#include "../common/Utils.h"

#include <chrono>
#include <iostream>


void performDataStandardizationOnCPU(std::vector<float>& data_host, unsigned number_of_columns, unsigned number_of_rows) {

    // start time measurement
    auto time_start = std::chrono::high_resolution_clock::now();

    // compute means and standard deviations for each data column
    std::vector<float> columns_means = computeColumnsMeansOnCPU(data_host, number_of_columns, number_of_rows);
    std::vector<float> columns_deviations = computeColumnsStandardDeviationsOnCPU(data_host, columns_means, number_of_columns, number_of_rows);

    // standardize data on CPU
    for(unsigned i = 0; i < number_of_rows; ++i) {
        unsigned data_index = i * number_of_columns;
        for(unsigned column = 0; column < number_of_columns; ++column) {
            data_host[data_index + column] = (data_host[data_index + column] - columns_means[column]) / columns_deviations[column];
        }
    }

    // finish time measurement, print the result
    auto time_stop = std::chrono::high_resolution_clock::now();
    std::cout << "[HOST] Data standardization - total elapsed time according to chrono lib: " << computeElapsedTimeInMilliseconds(time_start, time_stop) << " ms" << std::endl;

}