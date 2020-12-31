#ifndef CUDA_DATA_MINING_CPP_COMMANDLINEARGUMENTS_H
#define CUDA_DATA_MINING_CPP_COMMANDLINEARGUMENTS_H

#include <vector>
#include <string>


/**
 * Stores result of command line arguments parsing.
 */
struct CommandLineArguments {
    unsigned number_of_clusters {3};
    unsigned number_of_iterations {100};
    std::string input_file{};
    std::vector<unsigned> columns_to_omit;
};

#endif //CUDA_DATA_MINING_CPP_COMMANDLINEARGUMENTS_H
