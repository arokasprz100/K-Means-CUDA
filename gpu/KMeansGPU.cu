#include "KMeansGPU.h"

#include "CommonKernels.h"
#include "../common/Utils.h"
#include <float.h>


std::vector<unsigned> performKMeansOnGPU(float * data_device, unsigned number_of_rows, unsigned number_of_columns,
                                         unsigned number_of_clusters, const std::vector<unsigned>& initial_centroids,
                                         unsigned number_of_iterations) {

    // perform device query - get number of SMs, execution configuration
    unsigned number_of_multiprocessors = getNumberOfStreamingMultiprocessors();
    unsigned block_size = 512;
    unsigned number_of_blocks = number_of_multiprocessors * 2;
    std::cout << "[DEVICE] Execution configuration for KMeans: (blocks: " << number_of_blocks << ", threads: " << block_size << ")" << std::endl;

    // create time-measurement events
    cudaEvent_t time_start, time_stop;
    handleCudaErrors( cudaEventCreate(&time_start) );
    handleCudaErrors( cudaEventCreate(&time_stop) );

    // record the starting event
    handleCudaErrors( cudaEventRecord(time_start) );


    // allocate memory for centroids on GPU
    float * centroids_device = nullptr;
    unsigned centroids_memory_size = number_of_clusters * number_of_columns * sizeof(float);
    handleCudaErrors( cudaMalloc(&centroids_device, centroids_memory_size) );

    // copy initial centroids data using given indexes
    for(unsigned i = 0; i < number_of_clusters; ++i) {
        handleCudaErrors( cudaMemcpy(&centroids_device[i * number_of_columns], &data_device[initial_centroids.at(i) * number_of_columns], number_of_columns * sizeof(float), cudaMemcpyDeviceToDevice) );
    }

    // allocate memory for clusters count on GPU, do not initialize (will do it in each iteration)
    unsigned * clusters_count_device = nullptr;
    handleCudaErrors( cudaMalloc(&clusters_count_device, number_of_clusters * sizeof(unsigned)) );

    // allocate memory for closest cluster index on device, initialize to 0 (in case no iterations are performed)
    unsigned * closest_clusters_device = nullptr;
    handleCudaErrors( cudaMalloc(&closest_clusters_device, number_of_rows * sizeof(unsigned)) );
    handleCudaErrors( cudaMemset(closest_clusters_device, 0, number_of_rows * sizeof(unsigned)));

    // allocate memory for new centroids on GPU, do not initialize (no need)
    float * clusters_coordinates_sums = nullptr;
    handleCudaErrors( cudaMalloc(&clusters_coordinates_sums, centroids_memory_size) );

    // create special array of floats initialized to 0, that will be used to set centroids memory to zero
    // in each iteration using cudaMemset
    float * zero_filled_memory = nullptr;
    handleCudaErrors( cudaMalloc(&zero_filled_memory, centroids_memory_size) );
    zeroInitializeFloatMemory<<<1, 1>>>(zero_filled_memory, number_of_clusters * number_of_columns);
    handleCudaErrors( cudaGetLastError() );
    handleCudaErrors( cudaDeviceSynchronize() );


    // perform some number of iteration
    for(unsigned iteration = 0; iteration < number_of_iterations; ++iteration)
    {
        // compute distances from each point to each cluster, assign closest cluster to each point
        computeDistancesAndGetClosestClusters<<<number_of_blocks, block_size>>>(data_device, centroids_device,
                                                           number_of_rows, number_of_columns,
                                                           number_of_clusters, closest_clusters_device);
        handleCudaErrors( cudaGetLastError() );
        handleCudaErrors( cudaDeviceSynchronize() );


        // zero-initialize cluster coordinates sums and cluster count arrays (there will be addition performed there)
        handleCudaErrors( cudaMemcpy(clusters_coordinates_sums, zero_filled_memory, centroids_memory_size, cudaMemcpyDeviceToDevice) );
        handleCudaErrors( cudaMemset(clusters_count_device, 0, number_of_clusters * sizeof(unsigned)) );


        // sum coordinates of each centroid and count number of points for each centroid
        computeClustersCoordinatesSums<<<number_of_blocks, block_size>>>(data_device, clusters_coordinates_sums,
                                                    clusters_count_device, closest_clusters_device,
                                                    number_of_rows, number_of_columns);
        handleCudaErrors( cudaGetLastError() );
        handleCudaErrors( cudaDeviceSynchronize() );


        // compute new centroids using sums acquired in previous call
        computeNewCentroids<<<number_of_blocks, block_size>>>(centroids_device, clusters_coordinates_sums,
                                         clusters_count_device,
                                         number_of_columns, number_of_clusters);
        handleCudaErrors( cudaGetLastError() );
        handleCudaErrors( cudaDeviceSynchronize() );

    }

    // move clusters assignment to host
    std::vector<unsigned> clusters_assignment(number_of_rows);
    handleCudaErrors( cudaMemcpy(clusters_assignment.data(), closest_clusters_device, number_of_rows * sizeof(unsigned), cudaMemcpyDeviceToHost) );


    // record the stopping event and synchronize
    handleCudaErrors( cudaEventRecord(time_stop) );
    handleCudaErrors( cudaEventSynchronize(time_stop) );

    // compute and display elapsed time
    float elapsed_time = 0.0f;
    handleCudaErrors( cudaEventElapsedTime(&elapsed_time, time_start, time_stop) );
    std::cout << "[DEVICE] KMeans - total elapsed time according to CUDA events: " << elapsed_time << " ms" << std::endl;


    // destroy the event objects
    handleCudaErrors( cudaEventDestroy(time_start) );
    handleCudaErrors( cudaEventDestroy(time_stop) );

    // free GPU memory
    handleCudaErrors( cudaFree(closest_clusters_device) );
    handleCudaErrors( cudaFree(centroids_device) );

    // return vector with clusters numbers assigned to each dataset entry
    return clusters_assignment;
}


__global__ void computeDistancesAndGetClosestClusters(float * data, float * centroids,
                                                      unsigned input_size, unsigned row_size,
                                                      unsigned number_of_centroids,
                                                      unsigned * closest_clusters) {

    unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    // grid-stride loop
    for(unsigned i = start; i < input_size; i += stride) {
        unsigned data_index = i * row_size;

        float smallest_distance = FLT_MAX;
        unsigned closest_cluster = 0;

        // iterate over each centroid
        for(unsigned centroid = 0; centroid < number_of_centroids; ++centroid) {

            float distance = 0.0f;

            // iterate over each column in dataset
            for(unsigned column = 0; column < row_size; ++column) {
                distance += powf(centroids[centroid * row_size + column] - data[data_index + column], 2.0f);
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


__global__ void computeClustersCoordinatesSums(float * data, float * new_centroids_sums, unsigned * cluster_count,
                                               const unsigned * closest_clusters,
                                               unsigned number_of_rows, unsigned number_of_columns) {

    unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    for(unsigned i = start; i < number_of_rows; i += stride) {
        unsigned cluster_index = closest_clusters[i];
        unsigned data_index = i * number_of_columns;

        for(unsigned column = 0; column < number_of_columns; ++column) {
            atomicAdd(&new_centroids_sums[cluster_index * number_of_columns + column], data[data_index + column]);
        }
        atomicAdd(&cluster_count[cluster_index], 1);
    }

}


__global__ void computeNewCentroids(float * centroids, const float * new_centroids_sums,
                                    const unsigned * cluster_count,
                                    unsigned number_of_columns, unsigned number_of_clusters) {

    unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;

    for(unsigned i = start; i < number_of_clusters; i += stride) {
        unsigned index = i * number_of_columns;
        for(unsigned column = 0; column < number_of_columns; ++column) {
            if(cluster_count[i] != 0) {
                centroids[index + column] = new_centroids_sums[index + column] / static_cast<float>(cluster_count[i]);
            } // else no change, because the cluster is empty
        }
    }
}
