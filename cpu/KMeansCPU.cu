#include "KMeansCPU.h"

#include "../common/Utils.h"
#include <chrono>
#include <iostream>


std::vector<unsigned> performKMeansOnCPU(const std::vector<float>& data_host, unsigned number_of_rows, unsigned number_of_columns,
                                         unsigned number_of_clusters, const std::vector<unsigned>& initial_centroids,
                                         unsigned number_of_iterations) {

    // start time measurement
    auto time_start = std::chrono::high_resolution_clock::now();

    // create vector for centroids on host and copy initial centroids
    std::vector<float> centroids_host(number_of_clusters * number_of_columns);
    for(unsigned i = 0; i < number_of_clusters; ++i) {
        std::copy(&data_host[initial_centroids.at(i) * number_of_columns],
                  &data_host[initial_centroids.at(i) * number_of_columns] + number_of_columns,
                  &(centroids_host[i * number_of_columns]));
    }

    // allocate memory for closest cluster index on host
    std::vector<unsigned> closest_cluster_host(number_of_rows, 0);

    // allocate memory for clusters coordinates sums on host
    std::vector<float> clusters_coordinates_sums(number_of_clusters * number_of_columns, 0);

    // perform some number of iterations
    for(unsigned iteration = 0; iteration < number_of_iterations; ++iteration)
    {
        // compute distances from each point to each cluster, assign closest cluster to each point
        assignClosestClustersCPU(data_host, centroids_host, number_of_rows, number_of_columns, number_of_clusters, closest_cluster_host);

        // zero-initialize clusters coordinates sums
        std::fill(clusters_coordinates_sums.begin(), clusters_coordinates_sums.end(), 0.0f);

        // compute new centroids that will be used in a next iteration
        computeNewCentroidsCPU(data_host, clusters_coordinates_sums, centroids_host, closest_cluster_host, number_of_rows, number_of_columns, number_of_clusters);

    }

    // finish time measurement, print the result
    auto time_stop = std::chrono::high_resolution_clock::now();
    std::cout << "[HOST] KMeans - total elapsed time according to chrono lib: " << computeElapsedTimeInMilliseconds(time_start, time_stop) << " ms" << std::endl;


    // return vector with clusters numbers assigned to each dataset entry
    return closest_cluster_host;
}


void assignClosestClustersCPU(const std::vector<float>& data, const std::vector<float>& centroids,
                              unsigned number_of_rows, unsigned number_of_columns, unsigned number_of_centroids,
                              std::vector<unsigned>& closest_clusters) {

    for(unsigned i = 0; i < number_of_rows; ++i) {
        unsigned data_index = i * number_of_columns;

        float smallest_distance = std::numeric_limits<float>::infinity();
        unsigned closest_cluster = 0;

        // iterate over each centroid
        for(unsigned centroid = 0; centroid < number_of_centroids; ++centroid) {

            // compute distance from datapoint to given centroid
            float distance = 0.0f;
            for(unsigned column = 0; column < number_of_columns; ++column) {
                distance += powf(centroids[centroid * number_of_columns + column] - data[data_index + column], 2.0f);
            }
            distance = sqrt(distance);

            if(distance < smallest_distance) {
                smallest_distance = distance;
                closest_cluster = centroid;
            }
        }
        closest_clusters[i] = closest_cluster;
    }
}


void computeNewCentroidsCPU(const std::vector<float>& data,
                            std::vector<float>& cluster_coordinates_sums, std::vector<float>& centroids,
                            const std::vector<unsigned>& closest_clusters, unsigned number_of_rows, unsigned number_of_columns,
                            unsigned number_of_clusters) {

    std::vector<unsigned> cluster_count(number_of_clusters, 0);

    for(unsigned i = 0; i < number_of_rows; ++i) {
        unsigned cluster_index = closest_clusters[i];
        unsigned data_index = i * number_of_columns;
        for(unsigned column = 0; column < number_of_columns; ++column) {
            cluster_coordinates_sums[cluster_index * number_of_columns + column] += data[data_index + column];
        }
        ++cluster_count[cluster_index];
    }

    for(unsigned i = 0; i < number_of_clusters; ++i) {
        unsigned index = i * number_of_columns;
        for(unsigned column = 0; column < number_of_columns; ++column) {
            if (cluster_count[i] != 0) {
                centroids[index + column] = cluster_coordinates_sums[index + column] / static_cast<float>(cluster_count[i]);
            } // else not change, because the cluster is empty
        }
    }
}