#include <iostream>

#include "common/Dataset.h"
#include "common/CentroidsComputation.h"
#include "common/Utils.h"
#include "common/CommandLineArgumentsParser.h"

#include "gpu/KMeansGPU.h"
#include "gpu/StandardizationGPU.h"

#include "cpu/KMeansCPU.h"
#include "cpu/StandardizationCPU.h"

int main(int argc, char * argv[]) {

    // parse command line arguments
    auto settings = CommandLineArgumentsParser::parse(argc, argv);
    std::cout << "[COMMON] Number of clusters: " << settings.number_of_clusters << std::endl;
    std::cout << "[COMMON] Number of iterations: " << settings.number_of_iterations << std::endl;

    // data is stored using the row-major order
    std::cout << "[COMMON] Loading dataset from file " << settings.input_file << std::endl;
    auto dataset = Dataset::fromFile(settings.input_file, settings.columns_to_omit);

    // helper variables
    unsigned number_of_rows = dataset.getNumberOfRows();
    unsigned number_of_columns = dataset.getNumberOfColumns();
    unsigned total_number_of_elements = dataset.getNumberOfColumns() * dataset.getNumberOfRows();
    unsigned number_of_clusters = settings.number_of_clusters;
    unsigned number_of_iterations = settings.number_of_iterations;

    // generate initial centroids on CPU
    std::cout << "[COMMON] Generating initial centroids on CPU ... " << std::endl;
    auto initial_centroids = getRandomInitialCentroids(dataset.getNumberOfRows(), number_of_clusters);


    std::cout << std::endl;


    // print basic information about used GPU
    printInformationAboutGPUDevice();

    // allocate dataset memory on GPU, move dataset values to GPU
    std::cout << "[DEVICE] Moving dataset to GPU ... " << std::endl;
    float * data_device = nullptr;
    handleCudaErrors( cudaMalloc(&data_device, sizeof(float) * total_number_of_elements) );
    handleCudaErrors( cudaMemcpy(data_device, dataset.getMemory(), sizeof(float) * total_number_of_elements, cudaMemcpyHostToDevice) );

    // perform data standardization on GPU
    std::cout << "[DEVICE] Performing data standardization on GPU ... " << std::endl;
    performDataStandardizationOnGPU(data_device, number_of_columns, number_of_rows);

    // perform the algorithm on GPU
    std::cout << "[DEVICE] Performing K-Means algorithm on GPU ... " << std::endl;
    auto clusters_device = performKMeansOnGPU(data_device, number_of_rows, number_of_columns, number_of_clusters, initial_centroids, number_of_iterations);

    // save results to file
    std::cout << "[DEVICE] Saving GPU results to file ... " << std::endl;
    dataset.toFileWithClusters("../data/clusters_gpu.data", clusters_device);

    // free memory allocated for dataset on GPU
    handleCudaErrors( cudaFree(data_device) );


    std::cout << std::endl;


    // make copy of data on CPU
    std::cout << "[HOST] Creating copy of data on CPU ... " << std::endl;
    std::vector<float> data_host(dataset.getMemory(), dataset.getMemory() + total_number_of_elements);

    // perform data standardization on CPU
    std::cout << "[HOST] Performing data standardization on CPU ... " << std::endl;
    performDataStandardizationOnCPU(data_host, number_of_columns, number_of_rows);

    // perform the algorithm on CPU
    std::cout << "[HOST] Performing K-Means algorithm on CPU ... " << std::endl;
    auto clusters_host = performKMeansOnCPU(data_host, number_of_rows, number_of_columns, number_of_clusters, initial_centroids, number_of_iterations);

    // save results to file
    std::cout << "[HOST] Saving CPU results to file ..." << std::endl;
    dataset.toFileWithClusters("../data/clusters_cpu.data", clusters_host);


    std::cout << std::endl;


    // compare results of both algorithms
    std::cout << "[COMMON] Results comparison ... " << std::endl;
    std::vector<bool> identical_results(number_of_rows, true);
    for(unsigned row = 0; row < number_of_rows; ++row) {
        identical_results[row] = (clusters_host[row] == clusters_device[row]);
    }
    unsigned number_of_identical = std::count(identical_results.begin(), identical_results.end(), true);
    float percent_of_identical = 100.0f * static_cast<float>(number_of_identical) / static_cast<float>(number_of_rows);
    std::cout << "[COMMON] Number of matching results: " << number_of_identical << " (" << percent_of_identical << "%)" << std::endl;

    return 0;
}
