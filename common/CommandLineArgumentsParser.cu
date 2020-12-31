#include "CommandLineArgumentsParser.h"

#include <exception>
#include <stdexcept>


CommandLineArguments CommandLineArgumentsParser::parse(int argc, char * argv[]) {
    try
    {
        if(argc < 4) {
            throw std::runtime_error("[COMMON] Not enough input arguments: ./program n_clusters n_iter input_file <columns_to_skip...>");
        }
        CommandLineArguments arguments{};
        arguments.number_of_clusters = std::stoul(argv[1]);
        arguments.number_of_iterations = std::stoul(argv[2]);
        arguments.input_file = argv[3];

        unsigned argument_index = 4;
        while(argument_index < argc) {
            arguments.columns_to_omit.push_back(std::stoul(argv[argument_index]));
            ++argument_index;
        }
        return arguments;
    }
    catch(const std::exception& exception)
    {
        std::cout << "[COMMON] There was a problem when parsing command line arguments:\n" << exception.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}