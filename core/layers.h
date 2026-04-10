#pragma once

#include "matrix.h"

#include <cstdint>


namespace Neural {

enum class LayerType {
    Dense,
    Activation,
};

enum class ActivationType : std::uint8_t {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
};


template<typename T>
concept TLayer = requires(T layer, const Matrix& m) {
    { layer.forward(m) } -> std::same_as<Matrix>;
    { layer.backward(m) } -> std::same_as<Matrix>;
};

struct DenseLayer {
    Matrix weights;
    Matrix biases;
    Matrix inputCache;

    Matrix gradientWeights;
    Matrix gradiendBiases;

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& input);
    Matrix applyGradients(const double learningRate);
    void zeroGradients();
};

struct ActivationLayer {
    ActivationType activationType;

    Matrix inputCache;
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& output);
};

static_assert(TLayer<DenseLayer>, "Dense layer is a proper layer");
static_assert(TLayer<ActivationLayer>, "Activation layer is a proper layer");

}
