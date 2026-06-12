#pragma once

#include <filesystem>
#include <optional>
#include "core/lib/neural_network.h"

namespace Neural {

std::optional<NeuralNetwork> LoadNetwork(const std::filesystem::path& filename);
std::optional<NeuralNetworkConfiguration> LoadConfig(const std::filesystem::path& filename);
void SaveNetwork(const NeuralNetwork& network, const std::filesystem::path& filename);
NeuralNetwork CreateNetwork(const NeuralNetworkConfiguration& config);

}
