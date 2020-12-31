#include "CentroidsComputation.h"

#include <random>
#include <numeric>


std::vector<unsigned> getRandomInitialCentroids(unsigned number_of_rows, unsigned number_of_centroids) {
    std::vector<unsigned> dataset_indexes(number_of_rows);
    std::iota(dataset_indexes.begin(), dataset_indexes.end(), 0);
    std::shuffle(dataset_indexes.begin(), dataset_indexes.end(), std::mt19937(std::random_device()()));

    std::vector<unsigned> centroids;
    for(unsigned i = 0; i < number_of_centroids; ++i) {
        centroids.push_back(dataset_indexes.at(i));
    }
    return centroids;
}