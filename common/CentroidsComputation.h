#ifndef CUDA_DATA_MINING_CPP_CENTROIDSCOMPUTATION_H
#define CUDA_DATA_MINING_CPP_CENTROIDSCOMPUTATION_H

#include <vector>

/**
 * Generates indexes of initial centroids used by K-Means algorithm. Centroids are randomly chosen from the
 * dataset - generated numbers are the row numbers of centroids. Works on CPU.
 * @param number_of_rows Number of rows (entries, points) in the dataset.
 * @param number_of_centroids Number of centroids that should be chosen.
 * @return Vector with indexes of chosen centroids.
 */
std::vector<unsigned> getRandomInitialCentroids(unsigned number_of_rows, unsigned number_of_centroids);


#endif //CUDA_DATA_MINING_CPP_CENTROIDSCOMPUTATION_H
