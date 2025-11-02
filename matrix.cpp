#include "matrix.h"
#include <iostream>
#include <stdexcept>
#include <cassert>

#ifdef USE_EIGEN
#include "Eigen/Dense"
#endif

// Constructors
Matrix::Matrix() {
#ifdef USE_EIGEN
    data.resize(0, 0);
#else
    rows = 0;
    cols = 0;
#endif
}

Matrix::Matrix(size_t rows, size_t cols) {
#ifdef USE_EIGEN
    data.resize(rows, cols);
    data.setZero();
#else
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, std::vector<double>(cols, 0.0));
#endif
}

Matrix::Matrix(size_t rows, size_t cols, double initialValue) {
#ifdef USE_EIGEN
    data.resize(rows, cols);
    data.setConstant(initialValue);
#else
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, std::vector<double>(cols, initialValue));
#endif
}

Matrix::Matrix(const std::vector<std::vector<double>>& inputData) {
    if (inputData.empty()) {
#ifdef USE_EIGEN
        data.resize(0, 0);
#else
        rows = 0;
        cols = 0;
#endif
        return;
    }

    size_t input_rows = inputData.size();
    size_t input_cols = inputData[0].size();

    for (size_t i = 1; i < input_rows; ++i) {
        if (inputData[i].size() != input_cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }

#ifdef USE_EIGEN
    data.resize(input_rows, input_cols);
    for (size_t i = 0; i < input_rows; ++i) {
        for (size_t j = 0; j < input_cols; ++j) {
            data(i, j) = inputData[i][j];
        }
    }
#else
    rows = input_rows;
    cols = input_cols;
    data = inputData;
#endif
}

// Accessors
size_t Matrix::getRows() const {
#ifdef USE_EIGEN
    return data.rows();
#else
    return rows;
#endif
}

size_t Matrix::getCols() const {
#ifdef USE_EIGEN
    return data.cols();
#else
    return cols;
#endif
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
#ifdef USE_EIGEN
    return data(row, col);
#else
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
#endif
}

const double& Matrix::operator()(size_t row, size_t col) const {
#ifdef USE_EIGEN
    return data(row, col);
#else
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
#endif
}

// Arithmetic operations
Matrix Matrix::operator+(const Matrix& other) const {
#ifdef USE_EIGEN
    return Matrix(data + other.data);
#else
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
#endif
}

Matrix Matrix::operator-(const Matrix& other) const {
#ifdef USE_EIGEN
    return Matrix(data - other.data);
#else
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
#endif
}

Matrix Matrix::operator*(const Matrix& other) const {
#ifdef USE_EIGEN
    return Matrix(data * other.data);
#else
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
#endif
}

Matrix Matrix::operator*(double scalar) const {
#ifdef USE_EIGEN
    return Matrix(data * scalar);
#else
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
#endif
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }

#ifdef USE_EIGEN
    return Matrix(data / scalar);
#else
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] / scalar;
        }
    }
    return result;
#endif
}

Matrix Matrix::multiplyOptimized(const Matrix& other) const {
#ifdef USE_EIGEN
    // Eigen already uses optimized multiplication
    return *this * other;
#else
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
#endif
}

void Matrix::print() const {
#ifdef USE_EIGEN
    std::cout << data << std::endl;
#else
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
#endif
}

bool Matrix::isSquare() const {
#ifdef USE_EIGEN
    return data.rows() == data.cols();
#else
    return rows == cols;
#endif
}

Matrix Matrix::identity(size_t size) {
#ifdef USE_EIGEN
    return Matrix(Eigen::MatrixXd::Identity(size, size));
#else
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
#endif
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
#ifdef USE_EIGEN
    return Matrix(Eigen::MatrixXd::Zero(rows, cols));
#else
    return Matrix(rows, cols, 0.0);
#endif
}

Matrix Matrix::ones(size_t rows, size_t cols) {
#ifdef USE_EIGEN
    return Matrix(Eigen::MatrixXd::Ones(rows, cols));
#else
    return Matrix(rows, cols, 1.0);
#endif
}

Matrix Matrix::transpose() const {
#ifdef USE_EIGEN
    return Matrix(data.transpose());
#else
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
#endif
}

Matrix Matrix::transform(const size_t new_rows, const size_t new_cols) const {
#ifdef USE_EIGEN
    if (getRows() * getCols() != new_rows * new_cols) {
        throw std::invalid_argument("New size should have the same number of cells");
    }

    // Manually reshape in row-major order to match naive implementation
    Matrix result(new_rows, new_cols);
    size_t new_row = 0;
    size_t new_col = 0;
    for (size_t i = 0; i < getRows(); ++i) {
        for (size_t j = 0; j < getCols(); ++j) {
            result(new_row, new_col++) = data(i, j);
            if (new_col == new_cols) {
                new_col = 0;
                ++new_row;
            }
        }
    }
    return result;
#else
    if (cols * rows != new_rows * new_cols) {
        throw std::invalid_argument("New size should have the same number of cells");
    }
    Matrix result(new_rows, new_cols);
    size_t new_row = 0;
    size_t new_col = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(new_row, new_col++) = data[i][j];
            if (new_col == new_cols) {
                new_col = 0;
                ++new_row;
            }
        }
    }
    return result;
#endif
}
