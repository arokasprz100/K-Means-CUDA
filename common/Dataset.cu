#include "Dataset.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


Dataset Dataset::fromFile(const std::string& file_name, const std::vector<unsigned>& columns_to_omit) {
    try
    {
        std::ifstream data_file{file_name};
        if(!data_file.good()) {
            std::cout << "Could not open file: " << file_name << ". Aborting!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        std::vector<float> data;
        unsigned number_of_lines{0};
        while(std::getline(data_file, line)) {
            if(line.empty()) {
                continue;
            }

            ++number_of_lines;
            std::istringstream line_stream{line};
            std::string field;
            unsigned column_number{0};
            while (std::getline(line_stream, field, ',')) {
                if (std::find(columns_to_omit.begin(), columns_to_omit.end(), column_number) == columns_to_omit.end()) {
                    data.push_back(std::stof(field));
                }
                ++column_number;
            }
        }
        return Dataset(data, number_of_lines, data.size()/number_of_lines);
    }
    catch (const std::exception& exception)
    {
        std::cout << "[COMMON] There was a problem when parsing dataset file: " << exception.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}


void Dataset::toFileWithClusters(const std::string& file_name, const std::vector<unsigned>& clusters) const {
    std::ofstream output_file{file_name};
    for(unsigned row = 0; row < getNumberOfRows(); ++row) {
        for(unsigned column = 0; column < getNumberOfColumns(); ++column) {
            output_file << m_data[row * getNumberOfColumns() + column] << ",";
        }
        output_file << clusters[row] << std::endl;
    }
}