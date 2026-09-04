#pragma once

#include "matrix/matrix.h"

namespace Neural {

inline Matrix mse_gradient(const Matrix& output, const Matrix& target) {
    return output - target;
}

inline Matrix bce_gradient(const Matrix& output, const Matrix& target) {
    return output - target;
}

}
