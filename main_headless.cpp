#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

#include "digits_recognizer.h"


class InputParser{
    public:
        InputParser() = default;
        InputParser(const InputParser& other) : tokens(other.tokens) {}
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string& option, const std::string& defaultValue = {}) const {
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            return defaultValue;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};


bool validateDataset(const std::filesystem::path& datasetPath) {
    for (int i = 0; i < 10; ++i) {
        const auto subdirectory = datasetPath / std::to_string(i);
        if (!std::filesystem::exists(subdirectory) || !std::filesystem::is_directory(subdirectory)) {
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    InputParser cmd(argc, argv);

    if (!cmd.cmdOptionExists("--network")) {
        std::cerr << "Specify network name with --network first" << std::endl;
        return 1;
    }

    if (!cmd.cmdOptionExists("--dataset")) {
        std::cerr << "Specify dataset name with --dataset first" << std::endl;
        return 2;
    }

    const std::string networkName = cmd.getCmdOption("--network");
    const std::string datasetPath = cmd.getCmdOption("--dataset");
    if (!validateDataset(datasetPath)) {
        std::cerr << "Invalid dataset" << std::endl;
        return 3;
    }

    const size_t testingFileLimit = std::stoi(
        cmd.getCmdOption("--test-file-limit", "150")
    );
    const size_t datasetFileLimit = std::stoi(
        cmd.getCmdOption("dataset-file-limit", "10")
    );

    std::cout << "Starting with parameters:\n"
        << "\tNetwork name: " << networkName << "\n"
        << "\tDataset path: " << datasetPath << "\n"
        << "\tTesting file limit: " << testingFileLimit << "\n"
        << "\tDataset file limit: " << datasetFileLimit
        << std::endl;

    DigitsRecognizer recognizer;
    recognizer.loadNetwork(networkName);
    recognizer.setDataset(datasetPath);
    recognizer.setDatasetFileLimit(datasetFileLimit);
    recognizer.setTestingFileLimit(testingFileLimit);

    recognizer.doLearning();

    return 0;
}
