#include "core/generator/generator.h"

namespace Neural {

Generator::Generator(NeuralNetwork network, const std::size_t latentDim, const std::size_t numClasses)
    : applier(std::move(network)), latentDim(latentDim), numClasses(numClasses) {}

Matrix Generator::sampleNoise() const {
    std::normal_distribution<double> dist(0.0, 1.0);
    return Matrix(1, latentDim, [&](std::size_t, std::size_t) {
        return dist(rng);
    });
}

Matrix Generator::conditionedInput(const Matrix& noise, const std::size_t label) const {
    // Concatenate noise (1 x latentDim) with one-hot label (1 x numClasses)
    // to form a 1 x (latentDim + numClasses) input vector.
    return Matrix(1, latentDim + numClasses, [&](std::size_t, std::size_t col) -> double {
        if (col < latentDim)
            return noise(0, col);
        const std::size_t classIdx = col - latentDim;
        return (classIdx == label) ? 1.0 : 0.0;
    });
}

Matrix Generator::generate(const Matrix& noise, const std::size_t label) {
    return applier.forward(conditionedInput(noise, label), 0.0);
}

Matrix Generator::generate(const std::size_t label) {
    return generate(sampleNoise(), label);
}

Matrix Generator::backward(const Matrix& lossGradient) {
    return applier.backward(lossGradient);
}

void Generator::applyGradients(const double lr) {
    applier.applyGradients(lr);
}

void Generator::zeroGradients() {
    applier.zeroGradients();
}

std::size_t Generator::getLatentDim() const {
    return latentDim;
}

std::size_t Generator::getNumClasses() const {
    return numClasses;
}

const NeuralNetwork& Generator::getNetwork() const {
    return applier.getNeuralNetworkConfig();
}

}
