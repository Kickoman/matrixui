#ifdef USE_EIGEN

#include "matrix_impl.h"
#include "Eigen/Dense"
#include <iostream>
#include <stdexcept>
#include <memory>

class MatrixEigenImpl : public MatrixImpl {
private:
    Eigen::MatrixXd data;

public:
    MatrixEigenImpl() : data(0, 0) {}

    MatrixEigenImpl(size_t rows, size_t cols) : data(rows, cols) {
        data.setZero();
    }

    MatrixEigenImpl(size_t rows, size_t cols, double initialValue) : data(rows, cols) {
        data.setConstant(initialValue);
    }

    MatrixEigenImpl(const std::vector<std::vector<double>>& inputData) {
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

    MatrixEigenImpl(const Eigen::MatrixXd& eigenData) : data(eigenData) {}

    std::unique_ptr<MatrixImpl> clone() const override {
        return std::make_unique<MatrixEigenImpl>(data);
    }

    std::unique_ptr<MatrixImpl> create(size_t rows, size_t cols) const override {
        return std::make_unique<MatrixEigenImpl>(rows, cols);
    }

    std::unique_ptr<MatrixImpl> create(size_t rows, size_t cols, double initialValue) const override {
        return std::make_unique<MatrixEigenImpl>(rows, cols, initialValue);
    }

    std::unique_ptr<MatrixImpl> create(const std::vector<std::vector<double>>& data) const override {
        return std::make_unique<MatrixEigenImpl>(data);
    }

    size_t getRows() const override { return data.rows(); }
    size_t getCols() const override { return data.cols(); }

    double& operator()(size_t row, size_t col) override {
        return data(row, col);
    }

    const double& operator()(size_t row, size_t col) const override {
        return data(row, col);
    }

    std::unique_ptr<MatrixImpl> add(const MatrixImpl& other) const override {
        const MatrixEigenImpl& otherEigen = static_cast<const MatrixEigenImpl&>(other);
        return std::make_unique<MatrixEigenImpl>(data + otherEigen.data);
    }

    std::unique_ptr<MatrixImpl> subtract(const MatrixImpl& other) const override {
        const MatrixEigenImpl& otherEigen = static_cast<const MatrixEigenImpl&>(other);
        return std::make_unique<MatrixEigenImpl>(data - otherEigen.data);
    }

    std::unique_ptr<MatrixImpl> multiply(const MatrixImpl& other) const override {
        const MatrixEigenImpl& otherEigen = static_cast<const MatrixEigenImpl&>(other);
        return std::make_unique<MatrixEigenImpl>(data * otherEigen.data);
    }

    std::unique_ptr<MatrixImpl> multiply(double scalar) const override {
        return std::make_unique<MatrixEigenImpl>(data * scalar);
    }

    std::unique_ptr<MatrixImpl> divide(double scalar) const override {
        if (scalar == 0.0) {
            throw std::invalid_argument("Division by zero");
        }
        return std::make_unique<MatrixEigenImpl>(data / scalar);
    }

    std::unique_ptr<MatrixImpl> multiplyOptimized(const MatrixImpl& other) const override {
        // Eigen already uses optimized multiplication
        return multiply(other);
    }

    void print() const override {
        std::cout << data << std::endl;
    }

    bool isSquare() const override {
        return data.rows() == data.cols();
    }

    std::unique_ptr<MatrixImpl> identity(size_t size) const override {
        return std::make_unique<MatrixEigenImpl>(Eigen::MatrixXd::Identity(size, size));
    }

    std::unique_ptr<MatrixImpl> zeros(size_t rows, size_t cols) const override {
        return std::make_unique<MatrixEigenImpl>(Eigen::MatrixXd::Zero(rows, cols));
    }

    std::unique_ptr<MatrixImpl> ones(size_t rows, size_t cols) const override {
        return std::make_unique<MatrixEigenImpl>(Eigen::MatrixXd::Ones(rows, cols));
    }

    std::unique_ptr<MatrixImpl> transpose() const override {
        return std::make_unique<MatrixEigenImpl>(data.transpose());
    }

    std::unique_ptr<MatrixImpl> transform(size_t new_rows, size_t new_cols) const override {
        if (getRows() * getCols() != new_rows * new_cols) {
            throw std::invalid_argument("New size should have the same number of cells");
        }

        // Manually reshape in row-major order to match naive implementation
        MatrixEigenImpl result(new_rows, new_cols);
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
        return std::make_unique<MatrixEigenImpl>(std::move(result.data));
    }
};

// Factory function for Eigen implementation
std::unique_ptr<MatrixImpl> createMatrixImpl() {
    return std::make_unique<MatrixEigenImpl>();
}

#endif // USE_EIGEN
