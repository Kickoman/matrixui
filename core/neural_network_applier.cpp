#include "neural_network_applier.h"
#include "layers.h"
#include <stdexcept>

namespace Neural {

bool NeuralNetworkApplier::isInitialized() const {
    return initialized;
}

const NeuralNetwork& NeuralNetworkApplier::getNeuralNetworkConfig() const {
    return config;
}

void NeuralNetworkApplier::initializeNetwork(const NeuralNetwork& network) {
    config = network;
    initialized = true;
}

void NeuralNetworkApplier::initializeNetwork(NeuralNetwork&& network) {
    config = std::move(network);
    initialized = true;
}

NeuralNetworkApplier::NeuralNetworkApplier(const NeuralNetwork& network) {
    initializeNetwork(network);
}

NeuralNetworkApplier::NeuralNetworkApplier(NeuralNetwork&& network) {
    initializeNetwork(std::move(network));
}

Matrix NeuralNetworkApplier::predict(const Matrix& input) const {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initialized before prediction");
    }
    const ForwardContext ctx{false, 0.0};
    Matrix current = input;
    for (auto& layer : config.layerStack) {
        current = std::visit([&](auto& l) { return l.forward(current); }, layer);
    }
    return current;
}

Matrix NeuralNetworkApplier::train(
    const Matrix& input,
    const Matrix& target,
    const double learningRate,
    const double dropoutRate
) {
    for (auto& layer : config.layerStack) {
        std::visit([](auto& l) { l.zeroGradients(); }, layer);
    }

    const ForwardContext ctx{true, dropoutRate};
    Matrix current = input;
    for (auto& layer : config.layerStack) {
        current = std::visit([&](auto& l) { return l.forward(current, ctx); }, layer);
    }

    Matrix output = current;
    Matrix grad = current - target;

    for (int i = static_cast<int>(config.layerStack.size()) - 1; i >= 0; --i) {
        grad = std::visit([&](auto& l) { return l.backward(grad); }, config.layerStack[i]);
    }

    for (auto& layer : config.layerStack) {
        std::visit([lr = learningRate](auto& l) { l.applyGradients(lr); }, layer);
    }

    return output;
}

}
