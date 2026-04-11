#pragma once

#include <string>
#include <optional>
#include "neural_network.h"

namespace Neural {

std::optional<NeuralNetwork> LoadNetwork(const std::string& filename);
std::optional<NeuralNetworkConfiguration> LoadConfig(const std::string& filename);
void SaveNetwork(const NeuralNetwork& network, const std::string& filename);
NeuralNetwork CreateNetwork(const NeuralNetworkConfiguration& config);

}
