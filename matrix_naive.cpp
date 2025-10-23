#ifndef USE_EIGEN

#include "matrix_impl.h"
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <memory>

class MatrixNaiveImpl : public MatrixImpl {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    MatrixNaiveImpl() : rows(0), cols(0) {}

    MatrixNaiveImpl(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    MatrixNaiveImpl(size_t rows, size_t cols, double initialValue) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<double>(cols, initialValue));
    }

    MatrixNaiveImpl(const std::vector<std::vector<double>>& inputData) {
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

    std::unique_ptr<MatrixImpl> clone() const override {
        auto copy = std::make_unique<MatrixNaiveImpl>();
        copy->data = data;
        copy->rows = rows;
        copy->cols = cols;
        return copy;
    }

    std::unique_ptr<MatrixImpl> create(size_t rows, size_t cols) const override {
        return std::make_unique<MatrixNaiveImpl>(rows, cols);
    }

    std::unique_ptr<MatrixImpl> create(size_t rows, size_t cols, double initialValue) const override {
        return std::make_unique<MatrixNaiveImpl>(rows, cols, initialValue);
    }

    std::unique_ptr<MatrixImpl> create(const std::vector<std::vector<double>>& data) const override {
        return std::make_unique<MatrixNaiveImpl>(data);
    }

    size_t getRows() const override { return rows; }
    size_t getCols() const override { return cols; }

    double& operator()(size_t row, size_t col) override {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[row][col];
    }

    const double& operator()(size_t row, size_t col) const override {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[row][col];
    }

    std::unique_ptr<MatrixImpl> add(const MatrixImpl& other) const override {
        const MatrixNaiveImpl& otherNaive = static_cast<const MatrixNaiveImpl&>(other);
        if (rows != otherNaive.rows || cols != otherNaive.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        auto result = std::make_unique<MatrixNaiveImpl>(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*result)(i, j) = data[i][j] + otherNaive(i, j);
            }
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> subtract(const MatrixImpl& other) const override {
        const MatrixNaiveImpl& otherNaive = static_cast<const MatrixNaiveImpl&>(other);
        if (rows != otherNaive.rows || cols != otherNaive.cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }

        auto result = std::make_unique<MatrixNaiveImpl>(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*result)(i, j) = data[i][j] - otherNaive(i, j);
            }
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> multiply(const MatrixImpl& other) const override {
        const MatrixNaiveImpl& otherNaive = static_cast<const MatrixNaiveImpl&>(other);
        if (cols != otherNaive.rows) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }

        auto result = std::make_unique<MatrixNaiveImpl>(rows, otherNaive.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < otherNaive.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    (*result)(i, j) += data[i][k] * otherNaive(k, j);
                }
            }
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> multiply(double scalar) const override {
        auto result = std::make_unique<MatrixNaiveImpl>(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*result)(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> divide(double scalar) const override {
        if (scalar == 0.0) {
            throw std::invalid_argument("Division by zero");
        }

        auto result = std::make_unique<MatrixNaiveImpl>(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*result)(i, j) = data[i][j] / scalar;
            }
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> multiplyOptimized(const MatrixNaiveImpl& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }

        auto result = std::make_unique<MatrixNaiveImpl>(rows, other.cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                double temp = data[i][k];
                for (size_t j = 0; j < other.cols; ++j) {
                    (*result)(i, j) += temp * other(k, j);
                }
            }
        }

        return result;
    }

    std::unique_ptr<MatrixImpl> multiplyOptimized(const MatrixImpl& other) const override {
        const MatrixNaiveImpl& otherNaive = static_cast<const MatrixNaiveImpl&>(other);
        return multiplyOptimized(otherNaive);
    }

    void print() const override {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    bool isSquare() const override {
        return rows == cols;
    }

    std::unique_ptr<MatrixImpl> identity(size_t size) const override {
        auto result = std::make_unique<MatrixNaiveImpl>(size, size, 0.0);
        for (size_t i = 0; i < size; ++i) {
            (*result)(i, i) = 1.0;
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> zeros(size_t rows, size_t cols) const override {
        return std::make_unique<MatrixNaiveImpl>(rows, cols, 0.0);
    }

    std::unique_ptr<MatrixImpl> ones(size_t rows, size_t cols) const override {
        return std::make_unique<MatrixNaiveImpl>(rows, cols, 1.0);
    }

    std::unique_ptr<MatrixImpl> transpose() const override {
        auto result = std::make_unique<MatrixNaiveImpl>(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*result)(j, i) = data[i][j];
            }
        }
        return result;
    }

    std::unique_ptr<MatrixImpl> transform(size_t new_rows, size_t new_cols) const override {
        if (cols * rows != new_rows * new_cols) {
            throw std::invalid_argument("New size should have the same number of cells");
        }
        auto result = std::make_unique<MatrixNaiveImpl>(new_rows, new_cols);
        size_t new_row = 0;
        size_t new_col = 0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*result)(new_row, new_col++) = data[i][j];
                if (new_col == new_cols) {
                    new_col = 0;
                    ++new_row;
                }
            }
        }
        return result;
    }
};

// Factory function for naive implementation
std::unique_ptr<MatrixImpl> createMatrixImpl() {
    return std::make_unique<MatrixNaiveImpl>();
}

#endif // !USE_EIGEN
