#pragma once

#include "../matrix/matrix.h"

namespace Neural {

// Gradient of mean squared error loss w.r.t. network output.
// Use with any output activation.
inline Matrix mse_gradient(const Matrix& output, const Matrix& target) {
    return output - target;
}

// Gradient of binary cross-entropy loss w.r.t. network output.
// Uses the output - target shortcut, which is correct when the final layer is
// Sigmoid: the Sigmoid derivative cancels with the BCE denominator, so the
// combined gradient at the layer input is simply (output - target).
// Do NOT use this if the final layer is not Sigmoid.
inline Matrix bce_gradient(const Matrix& output, const Matrix& target) {
    return output - target;
}

}
