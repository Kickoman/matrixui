#pragma once

#include "core/classifier/learning_config.h"
#include "core/nn/neural_network.h"

#include <cstddef>
#include <string>

namespace ClassifierCli {

struct PredictOptions {
    std::string networkPath;
    std::string imagePath;
    std::size_t imageWidth{28};
    std::size_t imageHeight{28};
};

// The topology --layers and the activation flags start from, and what a
// --network-config file replaces wholesale. The activations come from
// NeuralNetworkConfiguration's own defaults (ReLU / Softmax).
inline Neural::NeuralNetworkConfiguration DefaultNetworkConfiguration() {
    Neural::NeuralNetworkConfiguration configuration{};
    configuration.layersSizes = {28 * 28, 50, 20, 10};
    return configuration;
}

struct TrainOptions {
    std::string networkPath;
    std::string workingDirectory{"training-data"};

    // Bound to --learning-config/--network-config so CLI11 validates and documents
    // them; the values are read straight out of argv by main.cpp before parsing.
    std::string learningConfigPath;
    std::string networkConfigPath;

    std::string datasetPath;
    std::string trainDatasetPath;
    std::string testDatasetPath;
    std::size_t testFileLimit{0};
    std::size_t imageWidth{28};
    std::size_t imageHeight{28};

    Neural::Classifier::LearningConfig learning{};
    Neural::NeuralNetworkConfiguration network = DefaultNetworkConfiguration();
};

}  // namespace ClassifierCli
