#include "matrix.h"

#ifdef USE_EIGEN
#include "Eigen/Dense"
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

Matrix Matrix::multiplyOptimized(const Matrix& other) const {
    return *this * other; // Eigen already uses optimized multiplication
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
#else // USE EIGEN
#include <iostream>
#include <stdexcept>
#include <cassert>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(size_t rows, size_t cols, double initialValue) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, initialValue));
}

Matrix::Matrix(size_t rows, size_t cols, std::function<double(size_t, size_t)> initialValueGenerator)
    : rows(rows)
    , cols(cols)
{
    data.resize(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = initialValueGenerator(i, j);
        }
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& inputData) {
    if (inputData.empty()) {
        rows = 0;
        cols = 0;
        return;
    }

    rows = inputData.size();
    cols = inputData[0].size();

    for (size_t i = 1; i < rows; ++i) {
        if (inputData[i].size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }

    data = inputData;
}

double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                result(i, j) += data[i][k] * other(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] / scalar;
        }
    }
    return result;
}

Matrix Matrix::multiplyOptimized(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix result(rows, other.cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < cols; ++k) {
            double temp = data[i][k];
            for (size_t j = 0; j < other.cols; ++j) {
                result(i, j) += temp * other(k, j);
            }
        }
    }

    return result;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::transform(const size_t rows, const size_t cols) const {
    if (this->cols * this->rows != rows * cols) {
        throw std::invalid_argument("New size should have the same number of cells");
    }
    Matrix result(rows, cols);
    size_t new_row = 0;
    size_t new_col = 0;
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result(new_row, new_col++) = data[i][j];
            if (new_col == cols) {
                new_col = 0;
                ++new_row;
            }
        }
    }
    return result;
}

#endif // USE EIGEN
