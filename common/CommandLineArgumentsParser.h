#ifndef CUDA_DATA_MINING_CPP_COMMANDLINEARGUMENTSPARSER_H
#define CUDA_DATA_MINING_CPP_COMMANDLINEARGUMENTSPARSER_H

#include "CommandLineArguments.h"

#include <string>
#include <iostream>


/**
 * Class responsible for parsing command line arguments.
 */
class CommandLineArgumentsParser {
public:

    /**
     * Parses command line arguments, returns struct with results.
     * @param argc Number of command line arguments
     * @param argv Command line arguments.
     * @return Struct with parsing results.
     */
    static CommandLineArguments parse(int argc, char * argv[]);

};

#endif //CUDA_DATA_MINING_CPP_COMMANDLINEARGUMENTSPARSER_H
