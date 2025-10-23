#ifndef MATRIX_IMPL_H
#define MATRIX_IMPL_H

#include <vector>
#include <memory>

class MatrixImpl {
public:
    virtual ~MatrixImpl() = default;

    // Virtual constructors
    virtual std::unique_ptr<MatrixImpl> clone() const = 0;
    virtual std::unique_ptr<MatrixImpl> create(size_t rows, size_t cols) const = 0;
    virtual std::unique_ptr<MatrixImpl> create(size_t rows, size_t cols, double initialValue) const = 0;
    virtual std::unique_ptr<MatrixImpl> create(const std::vector<std::vector<double>>& data) const = 0;

    // Accessors
    virtual size_t getRows() const = 0;
    virtual size_t getCols() const = 0;

    // Element access
    virtual double& operator()(size_t row, size_t col) = 0;
    virtual const double& operator()(size_t row, size_t col) const = 0;

    // Arithmetic operations
    virtual std::unique_ptr<MatrixImpl> add(const MatrixImpl& other) const = 0;
    virtual std::unique_ptr<MatrixImpl> subtract(const MatrixImpl& other) const = 0;
    virtual std::unique_ptr<MatrixImpl> multiply(const MatrixImpl& other) const = 0;
    virtual std::unique_ptr<MatrixImpl> multiply(double scalar) const = 0;
    virtual std::unique_ptr<MatrixImpl> divide(double scalar) const = 0;

    // Optimized multiplication
    virtual std::unique_ptr<MatrixImpl> multiplyOptimized(const MatrixImpl& other) const = 0;

    // Utility methods
    virtual void print() const = 0;
    virtual bool isSquare() const = 0;

    // Static factory methods
    virtual std::unique_ptr<MatrixImpl> identity(size_t size) const = 0;
    virtual std::unique_ptr<MatrixImpl> zeros(size_t rows, size_t cols) const = 0;
    virtual std::unique_ptr<MatrixImpl> ones(size_t rows, size_t cols) const = 0;

    // Matrix operations
    virtual std::unique_ptr<MatrixImpl> transpose() const = 0;
    virtual std::unique_ptr<MatrixImpl> transform(size_t rows, size_t cols) const = 0;
};

// Factory function to create the appropriate implementation
std::unique_ptr<MatrixImpl> createMatrixImpl();

#endif // MATRIX_IMPL_H
