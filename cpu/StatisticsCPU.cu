#include "StatisticsCPU.h"

#include "../common/Utils.h"
#include <chrono>
#include <iostream>


std::vector<float> computeColumnsMeansOnCPU(const std::vector<float>& data_host,
                                            unsigned number_of_columns, unsigned number_of_rows) {
    std::vector<float> columns_means(number_of_columns, 0);
    for(unsigned row = 0; row < number_of_rows; ++row) {
        for(unsigned column = 0; column < number_of_columns; ++column) {
            columns_means.at(column) += data_host.at(row * number_of_columns + column);
        }
    }
    for(auto& mean: columns_means) {
        mean = mean / static_cast<float>(number_of_rows);
    }
    return columns_means;
}


std::vector<float> computeColumnsStandardDeviationsOnCPU(const std::vector<float>& data_host,
                                                         const std::vector<float>& columns_means,
                                                         unsigned number_of_columns, unsigned number_of_rows) {

    std::vector<float> columns_deviations(number_of_columns, 0);
    for(unsigned row = 0; row < number_of_rows; ++row) {
        for(unsigned column = 0; column < number_of_columns; ++column) {
            columns_deviations.at(column) += powf(columns_means.at(column) - data_host.at(row * number_of_columns + column), 2.0);
        }
    }
    for(auto& deviation: columns_deviations) {
        deviation = sqrt(deviation / static_cast<float>(number_of_rows));
    }
    return columns_deviations;
}
