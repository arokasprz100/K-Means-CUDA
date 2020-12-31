cmake_minimum_required(VERSION 3.16)
project(CUDA_Data_Mining_cpp CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDA_Data_Mining_cpp
        main.cu
        gpu/KMeansGPU.h gpu/KMeansGPU.cu
        gpu/StandardizationGPU.h gpu/StandardizationGPU.cu
        gpu/StatisticsGPU.h
        gpu/CommonKernels.h gpu/CommonKernels.cu
        cpu/KMeansCPU.h cpu/KMeansCPU.cu
        cpu/StatisticsCPU.h cpu/StatisticsCPU.cu
        cpu/StandardizationCPU.h cpu/StandardizationCPU.cu
        common/Dataset.h common/Dataset.cu
        common/CentroidsComputation.h common/CentroidsComputation.cu
        common/Utils.h common/Utils.cu
        common/CommandLineArguments.h
        common/CommandLineArgumentsParser.h common/CommandLineArgumentsParser.cu)

set_target_properties(
        CUDA_Data_Mining_cpp
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)