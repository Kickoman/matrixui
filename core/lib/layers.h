#pragma once

#include "matrix/matrix.h"

#include <concepts>
#include <cstdint>
#include <variant>

#include <nlohmann/json.hpp>


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
concept TLayer = requires(T layer, const Matrix& m, Matrix& mmutable, double lr, const ForwardContext& ctx) {
    { const_cast<const T&>(layer).forward(m) } -> std::same_as<Matrix>;
    { const_cast<const T&>(layer).forward(m, ctx) } -> std::same_as<Matrix>;
    { layer.backward(m, m, m) } -> std::same_as<Matrix>;
    { layer.applyGradients(lr) };
    { layer.zeroGradients() };
};

struct DenseLayer {
    Matrix weights;
    Matrix biases;

    Matrix gradientWeights;
    Matrix gradientBiases;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx) const;
    Matrix backward(const Matrix& gradOutput, const Matrix& input, const Matrix& output);

    void applyGradients(double learningRate);
    void zeroGradients();
};

struct ActivationLayer {
    ActivationType activationType;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx) const;
    Matrix backward(const Matrix& gradOutput, const Matrix& input, const Matrix& output);

    void applyGradients(double) {}
    void zeroGradients() {}
};

struct DropoutLayer {
    mutable Matrix mask;

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx) const;
    Matrix backward(const Matrix& gradOutput, const Matrix& input, const Matrix& output);

    void applyGradients(double learningRate) {}
    void zeroGradients() {}
};

struct SoftmaxLayer {

    Matrix forward(const Matrix& input) const;
    Matrix forward(const Matrix& input, const ForwardContext& ctx) const;
    Matrix backward(const Matrix& gradOutput, const Matrix& input, const Matrix& output);

    void applyGradients(double learningRate) {}
    void zeroGradients() {}
};

static_assert(TLayer<DenseLayer>, "Dense layer is not a proper layer");
static_assert(TLayer<ActivationLayer>, "Activation layer is not a proper layer");
static_assert(TLayer<DropoutLayer>, "Dropout layer is not a proper layer");
static_assert(TLayer<SoftmaxLayer>, "Softmax layer is not a proper layer");

using LayerData = std::variant<
    DenseLayer,
    ActivationLayer,
    DropoutLayer,
    SoftmaxLayer
>;

NLOHMANN_JSON_SERIALIZE_ENUM(LayerType, {
    {LayerType::Dense, "dense"},
    {LayerType::Activation, "activation"},
});

NLOHMANN_JSON_SERIALIZE_ENUM(ActivationType, {
    {ActivationType::Sigmoid, "sigmoid"},
    {ActivationType::ReLU, "relu"},
    {ActivationType::Tanh, "tanh"},
    {ActivationType::Softmax, "softmax"},
    {ActivationType::LeakyReLU, "leakyrelu"},
});


}
