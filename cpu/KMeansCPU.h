#ifndef CUDA_DATA_MINING_CPP_KMEANSCPU_H
#define CUDA_DATA_MINING_CPP_KMEANSCPU_H

#include <vector>


/**
 * Performs K-Means clustering on a CPU.
 * @param data_host Data stored in row-major order in RAM memory.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_clusters Number of clusters that should be created by an algorithm.
 * @param initial_centroids Indexes of initial centroids.
 * @param number_of_iterations Number of iterations that should be performed. Set to 100 by default.
 * @return Vector with a cluster number assigned to each data entry.
 */
std::vector<unsigned> performKMeansOnCPU(const std::vector<float>& data_host,
                                         unsigned number_of_rows, unsigned number_of_columns,
                                         unsigned number_of_clusters, const std::vector<unsigned>& initial_centroids,
                                         unsigned number_of_iterations = 100);


/**
 * Computes distances between data points and centroids, assigns each data point to the closest centroid,
 * creating clusters in result.
 * @param data Data stored in row-major order in RAM memory.
 * @param centroids Centroids coordinates stored in row-major order (each row represents single centroid) in RAM.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_centroids Number of centroids/clusters.
 * @param closest_clusters Vector with memory allocated for clusters assignment, works as output argument.
 */
void assignClosestClustersCPU(const std::vector<float>& data, const std::vector<float>& centroids,
                              unsigned number_of_rows, unsigned number_of_columns,
                              unsigned number_of_centroids,
                              std::vector<unsigned>& closest_clusters);


/**
 * Computes new centroids on a CPU. Centroids coordinates are computes as means of coordinates of points that
 * are assigned to given cluster. The no points are assigned to cluster, centroid coordinates do not change.
 * @param data Data storied in row-major order in RAM memory.
 * @param cluster_coordinates_sums Vector with memory allocated and zeroed for coordinate sums. It should be reused.
 * @param centroids Coordinates of previous centroids. Will be replaced with new ones by this algorithm.
 * @param closest_clusters Clusters assigned to each data point.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_clusters Number of centroids/clusters in the dataset.
 */
void computeNewCentroidsCPU(const std::vector<float>& data,
                            std::vector<float>& cluster_coordinates_sums, std::vector<float>& centroids,
                            const std::vector<unsigned>& closest_clusters,
                            unsigned number_of_rows, unsigned number_of_columns,
                            unsigned number_of_clusters);


#endif //CUDA_DATA_MINING_CPP_KMEANSCPU_H
