## Overview 
This repository contains a simple implementations of *z-score* data standardization and *k-means* clustring algorithms created using CUDA framework and C++ language. Created program runs both parallel and single-threaded versions of aforementioned algorithms to compare the performance.

## Running the project
To build the project one should entry the project directory and execute following commands:
```bash
mkdir build
cd build
cmake ..
make
```
In addition, one can decide to build the project using either `Release` of `Debug` configuration. To achieve this, one should add `-DCMAKE_BUILD_TYPE=Release` or `-DCMAKE_BUILD_TYPE=Debug` option. To run this project, one should type:
```bash
./CUDA_Data_Mining_cpp <number_of_clusters> <number_of_iterations> <data_file> <columns_to_omit>
```
There are four command line arguments this program takes, but the last one could be optional, depending on used dataset. They have a following meaning:
 * `<number_of_clusters>` - number of clusters and initial centroids program should create
 * `<number_of_iterations>` - number of iteration of the K-Means algorithm that should be performed
 * `<data_file>` - path to file with data
 * `<colums_to_omit>` - indexes of dataset columns that should not be loaded from file (useful when the dataset contains non-numerical data, that can not be handled by this implementation)

For example, to create 3 clusters in 100 iteration using the *iris* dataset, one should type:
```bash
./CUDA_Data_Mining_cpp 3 100 ../data/iris.data 4
```

## Repository structure
This repository contains following directories and files:
 * `common` - contains C++ code responsible for common functionalities, for example: parsing command line arguments, loading dataset from file, handling CUDA errors and choosing initial centroids
 * `cpu` - contains C++ code with single-threaded implementation of data standardization and k-means algorithms
 * `gpu` - contains C++ code with CUDA implementation of data standardization and k-means algorithms
 * `python_scripts` - contains utility scripts for generating test data and plotting results
 * `data` - contains example dataset (Iris) used for testing
 * `main.cu` file - application entry point
 * `CMakeLists.txt` - CMake file that I used to build and test the project
