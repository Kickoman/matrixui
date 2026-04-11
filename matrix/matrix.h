#ifdef USE_EIGEN

#ifndef MATRIX_H
#define MATRIX_H

#include "Eigen/Dense"

class Matrix {
private:
    Eigen::MatrixXd data;

    Matrix(Eigen::MatrixXd&& data) : data(std::move(data)) {}
    Matrix(const Eigen::MatrixXd& data) : data(data) {}
public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initialValue);
    Matrix(size_t rows, size_t cols, std::function<double(size_t, size_t)> initialValueGenerator);
    Matrix(const std::vector<std::vector<double>>& data);

    size_t getRows() const { return data.rows(); }
    size_t getCols() const { return data.cols(); }

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    void setZero();

    Matrix multiplyOptimized(const Matrix& other) const; // Just calls operator*
    Matrix hadamard(const Matrix& other) const;

    void print() const;
    bool isSquare() const { return data.rows() == data.cols(); }

    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);

    Matrix transpose() const;
    Matrix transform(const size_t rows, const size_t cols) const;
};

std::ostream& operator<<(std::ostream& stream, const Matrix& m);

#endif // MATRIX_H

#else // USE_EIGEN

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cassert>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initialValue);
    Matrix(const std::vector<std::vector<double>>& data);

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    void setZero();

    Matrix multiplyOptimized(const Matrix& other) const;

    void print() const;
    bool isSquare() const { return rows == cols; }

    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);

    Matrix transpose() const;
    Matrix transform(const size_t rows, const size_t cols) const;
};

#endif // MATRIX_H
#endif // USE_EIGEN
