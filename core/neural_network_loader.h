#pragma once

#include <string>
#include "neural_network.h"

namespace Neural {

NeuralNetwork LoadNetwork(const std::string& filename, const std::vector<std::size_t>& layers);

void SaveNetwork(const NeuralNetwork& network, const std::string& filename);

}
