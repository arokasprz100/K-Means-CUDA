#ifndef CUDA_DATA_MINING_CPP_DATASET_H
#define CUDA_DATA_MINING_CPP_DATASET_H

#include <string>
#include <vector>


/**
 * Encapsulates the processed dataset (stored in RAM). Provides functionalities to load and save it.
 * Data is stored in a row-major order. Handles only numerical data (text columns should be skipped).
 */
class Dataset {
public:

    /**
     * Static factory method that produces the Dataset instance with data loaded from file.
     * Loads the dataset from given file. Data file should not contain header. Second argument specifies
     * indexes of columns that should not be loaded (for example text columns).
     * @param file_name Path to dataset file.
     * @param columns_to_omit Numbers of columns that should not be loaded.
     * @return Dataset object with loaded data.
     */
    static Dataset fromFile(const std::string& file_name, const std::vector<unsigned>& columns_to_omit);

    /**
     * Saves dataset to file with additional columns that denotes the clustering result.
     * @param file_name Name of resulting file.
     * @param clusters Clustering result.
     */
    void toFileWithClusters(const std::string& file_name, const std::vector<unsigned>& clusters) const;


    inline unsigned getNumberOfRows() const {
        return m_number_of_rows;
    }

    inline unsigned getNumberOfColumns() const {
        return m_number_of_columns;
    }

    inline const float* getMemory() const {
        return m_data.data();
    }

private:

    Dataset(std::vector<float> data, const unsigned number_of_rows, const unsigned number_of_columns)
            : m_data(std::move(data)), m_number_of_rows(number_of_rows), m_number_of_columns(number_of_columns) {}

    std::vector<float> m_data;
    unsigned m_number_of_rows;
    unsigned m_number_of_columns;

};



#endif //CUDA_DATA_MINING_CPP_DATASET_H
