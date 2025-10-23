#include "matrix.h"
#include "matrix_impl.h"
#include <stdexcept>

// Private implementation that directly uses the backend
class Matrix::MatrixImpl {
private:
    std::unique_ptr<::MatrixImpl> backendImpl;

public:
    MatrixImpl() : backendImpl(createMatrixImpl()) {}

    MatrixImpl(size_t rows, size_t cols) : backendImpl(createMatrixImpl()->create(rows, cols)) {}

    MatrixImpl(size_t rows, size_t cols, double initialValue) : backendImpl(createMatrixImpl()->create(rows, cols, initialValue)) {}

    MatrixImpl(const std::vector<std::vector<double>>& data) : backendImpl(createMatrixImpl()->create(data)) {}

    MatrixImpl(std::unique_ptr<::MatrixImpl> impl) : backendImpl(std::move(impl)) {}

    MatrixImpl(const MatrixImpl& other) : backendImpl(other.backendImpl->clone()) {}

    MatrixImpl(MatrixImpl&& other) noexcept = default;

    MatrixImpl& operator=(const MatrixImpl& other) {
        if (this != &other) {
            backendImpl = other.backendImpl->clone();
        }
        return *this;
    }

    MatrixImpl& operator=(MatrixImpl&& other) noexcept = default;

    size_t getRows() const { return backendImpl->getRows(); }
    size_t getCols() const { return backendImpl->getCols(); }

    double& operator()(size_t row, size_t col) { return (*backendImpl)(row, col); }
    const double& operator()(size_t row, size_t col) const { return (*backendImpl)(row, col); }

    MatrixImpl add(const MatrixImpl& other) const {
        return MatrixImpl(backendImpl->add(*other.backendImpl));
    }

    MatrixImpl subtract(const MatrixImpl& other) const {
        return MatrixImpl(backendImpl->subtract(*other.backendImpl));
    }

    MatrixImpl multiply(const MatrixImpl& other) const {
        return MatrixImpl(backendImpl->multiply(*other.backendImpl));
    }

    MatrixImpl multiply(double scalar) const {
        return MatrixImpl(backendImpl->multiply(scalar));
    }

    MatrixImpl divide(double scalar) const {
        return MatrixImpl(backendImpl->divide(scalar));
    }

    MatrixImpl multiplyOptimized(const MatrixImpl& other) const {
        return MatrixImpl(backendImpl->multiplyOptimized(*other.backendImpl));
    }

    void print() const { backendImpl->print(); }
    bool isSquare() const { return backendImpl->isSquare(); }

    MatrixImpl identity(size_t size) const {
        return MatrixImpl(backendImpl->identity(size));
    }

    MatrixImpl zeros(size_t rows, size_t cols) const {
        return MatrixImpl(backendImpl->zeros(rows, cols));
    }

    MatrixImpl ones(size_t rows, size_t cols) const {
        return MatrixImpl(backendImpl->ones(rows, cols));
    }

    MatrixImpl transpose() const {
        return MatrixImpl(backendImpl->transpose());
    }

    MatrixImpl transform(size_t rows, size_t cols) const {
        return MatrixImpl(backendImpl->transform(rows, cols));
    }
};

// Matrix class implementation
Matrix::Matrix() : pImpl(std::make_unique<MatrixImpl>()) {}

Matrix::Matrix(size_t rows, size_t cols) : pImpl(std::make_unique<MatrixImpl>(rows, cols)) {}

Matrix::Matrix(size_t rows, size_t cols, double initialValue) : pImpl(std::make_unique<MatrixImpl>(rows, cols, initialValue)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& data) : pImpl(std::make_unique<MatrixImpl>(data)) {}

Matrix::Matrix(const Matrix& other) : pImpl(std::make_unique<MatrixImpl>(*other.pImpl)) {}

Matrix::Matrix(Matrix&& other) noexcept : pImpl(std::move(other.pImpl)) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        pImpl = std::make_unique<MatrixImpl>(*other.pImpl);
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        pImpl = std::move(other.pImpl);
    }
    return *this;
}

Matrix::~Matrix() = default;

size_t Matrix::getRows() const { return pImpl->getRows(); }
size_t Matrix::getCols() const { return pImpl->getCols(); }

double& Matrix::operator()(size_t row, size_t col) { return (*pImpl)(row, col); }
const double& Matrix::operator()(size_t row, size_t col) const { return (*pImpl)(row, col); }

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->add(*other.pImpl));
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->subtract(*other.pImpl));
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->multiply(*other.pImpl));
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->multiply(scalar));
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->divide(scalar));
    return result;
}

Matrix Matrix::multiplyOptimized(const Matrix& other) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->multiplyOptimized(*other.pImpl));
    return result;
}

void Matrix::print() const { pImpl->print(); }
bool Matrix::isSquare() const { return pImpl->isSquare(); }

Matrix Matrix::identity(size_t size) {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(MatrixImpl().identity(size));
    return result;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(MatrixImpl().zeros(rows, cols));
    return result;
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(MatrixImpl().ones(rows, cols));
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->transpose());
    return result;
}

Matrix Matrix::transform(const size_t rows, const size_t cols) const {
    Matrix result;
    result.pImpl = std::make_unique<MatrixImpl>(pImpl->transform(rows, cols));
    return result;
}
