#pragma once

#include <iosfwd>
#include <optional>
#include "core/nn/neural_network.h"

namespace Neural {

NeuralNetwork LoadNetwork(std::istream& in);
std::optional<NeuralNetworkConfiguration> LoadConfig(std::istream& in);
void SaveNetwork(std::ostream& out, const NeuralNetwork& network);
NeuralNetwork CreateNetwork(const NeuralNetworkConfiguration& config);

}
