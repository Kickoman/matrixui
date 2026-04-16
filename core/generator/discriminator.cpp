#include "core/generator/discriminator.h"

namespace Neural {

Discriminator::Discriminator(NeuralNetwork network)
    : applier(std::move(network)) {}

Matrix Discriminator::score(const Matrix& input) const {
    return applier.predict(input);
}

Matrix Discriminator::forward(const Matrix& input, const double dropoutRate) {
    return applier.forward(input, dropoutRate);
}

Matrix Discriminator::backward(const Matrix& lossGradient) {
    return applier.backward(lossGradient);
}

void Discriminator::applyGradients(const double lr) {
    applier.applyGradients(lr);
}

void Discriminator::zeroGradients() {
    applier.zeroGradients();
}

bool Discriminator::isInitialized() const {
    return applier.isInitialized();
}

const NeuralNetwork& Discriminator::getNetwork() const {
    return applier.getNeuralNetworkConfig();
}

}
