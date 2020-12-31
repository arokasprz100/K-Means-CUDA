#ifndef CUDA_DATA_MINING_CPP_KMEANSGPU_H
#define CUDA_DATA_MINING_CPP_KMEANSGPU_H

#include <vector>


/**
 * Performs K-Means clustering on GPU.
 * @param data_device Pointer to data stored in GPU memory in row-major order (each row is a single data point).
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_clusters Number of clusters that should be created by this algorithm.
 * @param initial_centroids Indexes of initial centroids generated outside of this function.
 * @param number_of_iterations Number of algorithm iterations that should be performed. Set to 100 by default.
 * @return Vector (in RAM memory) with a cluster number assigned to each data entry.
 */
std::vector<unsigned> performKMeansOnGPU(float * data_device,
                                         unsigned number_of_rows, unsigned number_of_columns,
                                         unsigned number_of_clusters, const std::vector<unsigned>& initial_centroids,
                                         unsigned number_of_iterations = 100);


/**
 * GPU kernel. Computes distances between data points and centroids, assigns each data point to the closest
 * centroid, creating clusters in result.
 * @param data Pointer to GPU memory that stored data in a row-major order.
 * @param centroids Pointer to GPU memory that stores centroids in a row-major order (each row represents centroid).
 * @param input_size Number of rows (entries) in the dataset.
 * @param row_size Number of columns in the dataset.
 * @param number_of_centroids Number of clusters/centroids.
 * @param closest_clusters GPU memory allocated for clusters assignment, works as output argument.
 */
__global__
void computeDistancesAndGetClosestClusters(float * data, float * centroids,
                                           unsigned input_size, unsigned row_size,
                                           unsigned number_of_centroids, unsigned * closest_clusters);


/**
 * GPU kernel. Computes sums of coordinates of points that are assigned to each cluster. Counts points
 * assigned to each cluster.
 * @param data Pointer to GPU memory that stored data in a row-major order.
 * @param new_centroids_sums Zero-initialized GPU memory allocated to store sums of centroids coordinates.
 * @param cluster_count Zero-initialized GPU memory allocated to store number of elements in each cluster.
 * @param closest_clusters Indexes of clusters assigned to each data point.
 * @param number_of_rows Number of rows (entries) in the dataset.
 * @param number_of_columns Number of columns in the dataset.
 */
__global__
void computeClustersCoordinatesSums(float * data, float * new_centroids_sums, unsigned * cluster_count,
                                    const unsigned * closest_clusters,
                                    unsigned number_of_rows, unsigned number_of_columns);


/**
 * Computes centroid coordinates by performing division of coordinates sums by the number of entries assigned to
 * each cluster.
 * @param centroids Pointer to GPU memory with coordinated of old centroids that will be replaced with new ones.
 * @param new_centroids_sums GPU memory that stored sums of coordinates of points assigned to each cluster.
 * @param cluster_count GPU memory that stores number of entries assigned to each cluster.
 * @param number_of_columns Number of columns in the dataset.
 * @param number_of_clusters Number of cluster/centroids.
 */
__global__
void computeNewCentroids(float * centroids, const float * new_centroids_sums,
                         const unsigned * cluster_count,
                         unsigned number_of_columns, unsigned number_of_clusters);


#endif //CUDA_DATA_MINING_CPP_KMEANSGPU_H
