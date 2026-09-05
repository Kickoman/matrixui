#include "core/nn/neural_network_applier.h"
#include "core/nn/layers.h"
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

    activations.resize(config.layerStack.size() + 1);
}

void NeuralNetworkApplier::initializeNetwork(NeuralNetwork&& network) {
    config = std::move(network);
    initialized = true;

    activations.resize(config.layerStack.size() + 1);
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
        current = std::visit([&](auto& l) { return l.forward(current, ctx); }, layer);
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
    activations[0] = input;

    const auto layersCount = config.layerStack.size();
    for (size_t i = 0; i < layersCount; ++i) {
        const auto& layer = config.layerStack[i];
        activations[i + 1] = std::visit([&](auto& l) { return l.forward(activations[i], ctx); }, layer);
    }

    return activations.back();
}

Matrix NeuralNetworkApplier::backward(const Matrix& lossGradient) {
    Matrix grad = lossGradient;
    std::int64_t layersSize = static_cast<int>(config.layerStack.size());
    for (int i = layersSize - 1; i >= 0; --i) {
        grad = std::visit([&](auto& l) { return l.backward(grad, activations[i], activations[i + 1]); }, config.layerStack[i]);
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
