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

void NeuralNetworkApplier::zeroGradients() {
    for (auto& layer : config.layerStack) {
        std::visit([](auto& l) { l.zeroGradients(); }, layer);
    }
}

Matrix NeuralNetworkApplier::forward(const Matrix& input, const double dropoutRate) {
    if (!isInitialized()) {
        throw std::runtime_error("Network must be initialized before forward pass");
    }
    const ForwardContext ctx{true, dropoutRate};
    Matrix current = input;
    for (auto& layer : config.layerStack) {
        current = std::visit([&](auto& l) { return l.forward(current, ctx); }, layer);
    }
    return current;
}

Matrix NeuralNetworkApplier::backward(const Matrix& lossGradient) {
    Matrix grad = lossGradient;
    for (int i = static_cast<int>(config.layerStack.size()) - 1; i >= 0; --i) {
        grad = std::visit([&](auto& l) { return l.backward(grad); }, config.layerStack[i]);
    }
    return grad;
}

void NeuralNetworkApplier::applyGradients(const double learningRate) {
    for (auto& layer : config.layerStack) {
        std::visit([lr = learningRate](auto& l) { l.applyGradients(lr); }, layer);
    }
}

Matrix NeuralNetworkApplier::train(
    const Matrix& input,
    const Matrix& target,
    const double learningRate,
    const double dropoutRate
) {
    zeroGradients();
    const Matrix output = forward(input, dropoutRate);
    backward(output - target);
    applyGradients(learningRate);
    return output;
}

}
