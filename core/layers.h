#pragma once

#include "matrix.h"

#include <concepts>
#include <cstdint>
#include <variant>


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
    LeakyReLU,
};

struct ForwardContext {
    bool training;
    double dropoutRate;
};

template<typename T>
concept TLayer = requires(T layer, const Matrix& m, double lr, const ForwardContext& ctx) {
    { const_cast<const T&>(layer).forward(m) } -> std::same_as<Matrix>;
    { layer.forward(m, ctx) } -> std::same_as<Matrix>;
    { layer.backward(m) } -> std::same_as<Matrix>;
    { layer.applyGradients(lr) };
    { layer.zeroGradients() };
};

struct DenseLayer {
    Matrix weights;
    Matrix biases;

    Matrix inputCache;

    Matrix gradientWeights;
    Matrix gradientBiases;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx);
    Matrix backward(const Matrix& gradOutput);

    void applyGradients(double learningRate);
    void zeroGradients();
};

struct ActivationLayer {
    ActivationType activationType;

    Matrix outputCache;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx);
    Matrix backward(const Matrix& gradOutput);

    void applyGradients(double) {}
    void zeroGradients() {}
};

struct DropoutLayer {
    Matrix mask;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx);
    Matrix backward(const Matrix& gradOutput);

    void applyGradients(double learningRate) {}
    void zeroGradients() {}
};

struct SoftmaxLayer {
    Matrix outputCache;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx);
    Matrix backward(const Matrix& gradOutput);

    void applyGradients(double learningRate) {}
    void zeroGradients() {}
};

static_assert(TLayer<DenseLayer>, "Dense layer is a proper layer");
static_assert(TLayer<ActivationLayer>, "Activation layer is a proper layer");
static_assert(TLayer<DropoutLayer>, "Dropout layer is a proper layer");
static_assert(TLayer<SoftmaxLayer>, "Softmax layer is a proper layer");

using LayerData = std::variant<
    DenseLayer,
    ActivationLayer,
    DropoutLayer,
    SoftmaxLayer
>;

}
