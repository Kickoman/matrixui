#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cassert>

#ifdef USE_EIGEN
#include "Eigen/Dense"
#endif

class Matrix {
private:
#ifdef USE_EIGEN
    Eigen::MatrixXd data;

    // Private constructors for Eigen backend
    Matrix(Eigen::MatrixXd&& eigenData) : data(std::move(eigenData)) {}
    Matrix(const Eigen::MatrixXd& eigenData) : data(eigenData) {}
#else
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;
#endif

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initialValue);
    Matrix(const std::vector<std::vector<double>>& data);

    // Copy and Move operations
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) noexcept = default;
    ~Matrix() = default;

    // Accessors
    size_t getRows() const;
    size_t getCols() const;

    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    // Arithmetic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    // Optimized multiplication
    Matrix multiplyOptimized(const Matrix& other) const;

    // Utility methods
    void print() const;
    bool isSquare() const;

    // Static factory methods
    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);

    // Matrix operations
    Matrix transpose() const;
    Matrix transform(const size_t rows, const size_t cols) const;
};

#endif // MATRIX_H
