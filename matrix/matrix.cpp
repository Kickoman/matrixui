#include "matrix/matrix.h"

#include <iostream>
#include <stdexcept>

Matrix::Matrix() : data(0, 0) {}

Matrix::Matrix(size_t rows, size_t cols) : data(rows, cols) {
    data.setZero();
}

Matrix::Matrix(size_t rows, size_t cols, double initialValue) : data(rows, cols) {
    data.setConstant(initialValue);
}

Matrix::Matrix(size_t rows, size_t cols, std::function<double(size_t, size_t)> initialValueGenerator)
    : data(rows, cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data(i, j) = initialValueGenerator(i, j);
        }
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& inputData) {
    if (inputData.empty()) {
        data.resize(0, 0);
        return;
    }

    size_t rows = inputData.size();
    size_t cols = inputData[0].size();

    for (size_t i = 1; i < rows; ++i) {
        if (inputData[i].size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }

    data.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data(i, j) = inputData[i][j];
        }
    }
}

double& Matrix::operator()(size_t row, size_t col) {
    return data(row, col);
}

const double& Matrix::operator()(size_t row, size_t col) const {
    return data(row, col);
}

Matrix Matrix::operator+(const Matrix& other) const {
    return Matrix(data + other.data);
}

Matrix Matrix::operator-(const Matrix& other) const {
    return Matrix(data - other.data);
}

Matrix Matrix::operator*(const Matrix& other) const {
    return Matrix(data * other.data);
}

Matrix Matrix::operator*(double scalar) const {
    return Matrix(data * scalar);
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return Matrix(data / scalar);
}

void Matrix::setZero() {
    data.setZero();
}

Matrix& Matrix::operator+=(const Matrix& other) {
    data += other.data;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    data -= other.data;
    return *this;
}

Matrix Matrix::multiplyOptimized(const Matrix& other) const {
    return *this * other; // Eigen already uses optimized multiplication
}

Matrix Matrix::hadamard(const Matrix& other) const {
    assert(other.getCols() == this->getCols());
    assert(other.getRows() == this->getRows());
    return Matrix(data.cwiseProduct(other.data));
}

void Matrix::print() const {
    std::cout << data << std::endl;
}

Matrix Matrix::identity(size_t size) {
    return Matrix(Eigen::MatrixXd::Identity(size, size));
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(Eigen::MatrixXd::Zero(rows, cols));
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(Eigen::MatrixXd::Ones(rows, cols));
}

Matrix Matrix::transpose() const {
    return Matrix(data.transpose());
}

Matrix Matrix::transform(const size_t rows, const size_t cols) const {
    if (getRows() * getCols() != rows * cols) {
        throw std::invalid_argument("New size should have the same number of cells");
    }

    // Manually reshape in row-major order to match naive implementation
    Matrix result(rows, cols);
    size_t new_row = 0;
    size_t new_col = 0;
    for (size_t i = 0; i < getRows(); ++i) {
        for (size_t j = 0; j < getCols(); ++j) {
            result(new_row, new_col++) = data(i, j);
            if (new_col == cols) {
                new_col = 0;
                ++new_row;
            }
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& stream, const Matrix& m) {
    for (std::size_t row = 0; row < m.getRows(); ++row) {
        for (std::size_t col = 0; col < m.getCols(); ++col) {
            stream << m(row, col) << " ";
        }
        stream << "\n";
    }
    return stream;
}
