#include "core/nn/layers.h"
#include <random>


namespace Neural {


static constexpr double kLeakyReLUAlpha = 0.2;

double applyActivation(const double x, const ActivationType type) {
    assert(type != ActivationType::Softmax);
    switch(type) {
        case ActivationType::Sigmoid:   return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU:      return std::max(0.0, x);
        case ActivationType::LeakyReLU: return x > 0.0 ? x : kLeakyReLUAlpha * x;
        case ActivationType::Tanh:      return std::tanh(x);
        default: return x;
    }
}

double applyActivationDerivative(const double x, const ActivationType type) {
    assert(type != ActivationType::Softmax);
    switch(type) {
        case ActivationType::Sigmoid:   return x * (1.0 - x);
        case ActivationType::ReLU:      return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::LeakyReLU: return x > 0.0 ? 1.0 : kLeakyReLUAlpha;
        case ActivationType::Tanh:      return 1.0 - x*x;
        default: return 1.0;
    }
}

void ApplySoftMax(Matrix& m) {
    for (std::size_t row = 0; row < m.getRows(); ++row) {
        double maxValue = m(row, 0);
        for (std::size_t col = 1; col < m.getCols(); ++col) {
            maxValue = std::max(maxValue, m(row, col));
        }

        double sum = 0;
        for (std::size_t col = 0; col < m.getCols(); ++col) {
            m(row, col) = std::exp(m(row, col) - maxValue);
            sum += m(row, col);
        }

        for (std::size_t col = 0; col < m.getCols(); ++col) {
            m(row, col) /= sum;
        }
    }
}

Matrix DenseLayer::forward(const Matrix& input) const {
    return input.multiplyAdd(weights, biases);
}

Matrix DenseLayer::forward(const Matrix& input, const ForwardContext&) const {
    return forward(input);
}

Matrix DenseLayer::backward(const Matrix& gradOutput, const Matrix& input, const Matrix& /* output */) {
    gradientWeights.addTransposeMultiply(input, gradOutput);
    gradientBiases += gradOutput;
    return gradOutput.multiplyTranspose(weights);
}

void DenseLayer::applyGradients(const double learningRate) {
    weights.substractScaled(learningRate, gradientWeights);
    biases.substractScaled(learningRate, gradientBiases);
}

void DenseLayer::zeroGradients() {
    gradientWeights.setZero();
    gradientBiases.setZero();
}

Matrix ActivationLayer::forward(const Matrix& input) const {
    Matrix activation = input;
    for (std::size_t row = 0; row < activation.getRows(); ++row) {
        for (std::size_t col = 0; col < activation.getCols(); ++col) {
            activation(row, col) = applyActivation(input(row, col), activationType);
        }
    }
    return activation;
}

Matrix ActivationLayer::forward(const Matrix& input, const ForwardContext&) const {
    return forward(input);
}

Matrix ActivationLayer::backward(const Matrix& gradOutput, const Matrix& input, const Matrix& output) {
    Matrix gradient = gradOutput;

    const auto& activation =
        activationType == ActivationType::ReLU || activationType == ActivationType::LeakyReLU
        ? input
        : output;

    for (std::size_t row = 0; row < gradient.getRows(); ++row) {
        for (std::size_t col = 0; col < gradient.getCols(); ++col) {
            gradient(row, col) *= applyActivationDerivative(activation(row, col), activationType);
        }
    }
    return gradient;
}


Matrix DropoutLayer::forward(const Matrix& input) const {
    return input;
}

Matrix DropoutLayer::forward(const Matrix& input, const ForwardContext& ctx) const {
    if (!ctx.training || ctx.dropoutRate == 0) {
        return input;
    }

    static std::mt19937 generator(std::random_device{}());
    std::bernoulli_distribution distribution(1. - ctx.dropoutRate);

    Matrix output = input;
    mask = Matrix::ones(input.getRows(), input.getCols());

    for (std::size_t row = 0; row < input.getRows(); ++row) {
        for (std::size_t col = 0; col < input.getCols(); ++col) {
            const bool shouldDrop = !distribution(generator);
            if (shouldDrop) {
                output(row, col) = 0;
                mask(row, col) = 0;
            } else {
                output(row, col) /= (1. - ctx.dropoutRate);
            }
        }
    }

    return output;
}

Matrix DropoutLayer::backward(const Matrix& gradOutput, const Matrix& /* input */, const Matrix& /* output */) {
    if (mask.getRows() == 0) {
        return gradOutput;
    }
    return gradOutput.hadamard(mask);
}


Matrix SoftmaxLayer::forward(const Matrix& input) const {
    Matrix out = input;
    ApplySoftMax(out);
    return out;
}

Matrix SoftmaxLayer::forward(const Matrix& input, const ForwardContext&) const {
    return forward(input);
}

Matrix SoftmaxLayer::backward(const Matrix& gradOutput, const Matrix& /*input*/, const Matrix& /* output */) {
    return gradOutput;
}

}
