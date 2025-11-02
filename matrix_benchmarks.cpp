#include "matrix.h"
#include "perftest/include/perfbench/benchmark.h"
#include "perftest/include/perfbench/registry.h"
#include <vector>
#include <random>

// Benchmark: Matrix multiplication
class MatrixMultiplicationBenchmark : public perfbench::Benchmark {
public:
    MatrixMultiplicationBenchmark() : Benchmark("MatrixMultiplication") {
        setDescription("Matrix multiplication of two 100x100 matrices");
    }

    void setUp() override {
        // Create two 100x100 matrices with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        matrixA = Matrix(100, 100);
        matrixB = Matrix(100, 100);

        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < 100; ++j) {
                matrixA(i, j) = dis(gen);
                matrixB(i, j) = dis(gen);
            }
        }
    }

    void run() override {
        result = matrixA * matrixB;
        // Prevent optimization
        volatile double dummy = result(0, 0);
    }

    void tearDown() override {
        // Cleanup
    }

private:
    Matrix matrixA;
    Matrix matrixB;
    Matrix result;
};

// Benchmark: Optimized matrix multiplication
class MatrixOptimizedMultiplicationBenchmark : public perfbench::Benchmark {
public:
    MatrixOptimizedMultiplicationBenchmark() : Benchmark("MatrixOptimizedMultiplication") {
        setDescription("Optimized matrix multiplication of two 100x100 matrices");
    }

    void setUp() override {
        // Create two 100x100 matrices with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        matrixA = Matrix(100, 100);
        matrixB = Matrix(100, 100);

        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < 100; ++j) {
                matrixA(i, j) = dis(gen);
                matrixB(i, j) = dis(gen);
            }
        }
    }

    void run() override {
        result = matrixA.multiplyOptimized(matrixB);
        // Prevent optimization
        volatile double dummy = result(0, 0);
    }

    void tearDown() override {
        // Cleanup
    }

private:
    Matrix matrixA;
    Matrix matrixB;
    Matrix result;
};

// Benchmark: Matrix addition
class MatrixAdditionBenchmark : public perfbench::Benchmark {
public:
    MatrixAdditionBenchmark() : Benchmark("MatrixAddition") {
        setDescription("Matrix addition of two 200x200 matrices");
    }

    void setUp() override {
        // Create two 200x200 matrices with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        matrixA = Matrix(200, 200);
        matrixB = Matrix(200, 200);

        for (size_t i = 0; i < 200; ++i) {
            for (size_t j = 0; j < 200; ++j) {
                matrixA(i, j) = dis(gen);
                matrixB(i, j) = dis(gen);
            }
        }
    }

    void run() override {
        result = matrixA + matrixB;
        // Prevent optimization
        volatile double dummy = result(0, 0);
    }

    void tearDown() override {
        // Cleanup
    }

private:
    Matrix matrixA;
    Matrix matrixB;
    Matrix result;
};

// Benchmark: Matrix transpose
class MatrixTransposeBenchmark : public perfbench::Benchmark {
public:
    MatrixTransposeBenchmark() : Benchmark("MatrixTranspose") {
        setDescription("Matrix transpose of a 300x300 matrix");
    }

    void setUp() override {
        // Create a 300x300 matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        matrix = Matrix(300, 300);

        for (size_t i = 0; i < 300; ++i) {
            for (size_t j = 0; j < 300; ++j) {
                matrix(i, j) = dis(gen);
            }
        }
    }

    void run() override {
        result = matrix.transpose();
        // Prevent optimization
        volatile double dummy = result(0, 0);
    }

    void tearDown() override {
        // Cleanup
    }

private:
    Matrix matrix;
    Matrix result;
};

// Benchmark: Matrix scalar multiplication
class MatrixScalarMultiplicationBenchmark : public perfbench::Benchmark {
public:
    MatrixScalarMultiplicationBenchmark() : Benchmark("MatrixScalarMultiplication") {
        setDescription("Matrix scalar multiplication of a 200x200 matrix");
    }

    void setUp() override {
        // Create a 200x200 matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        matrix = Matrix(200, 200);
        scalar = 2.5;

        for (size_t i = 0; i < 200; ++i) {
            for (size_t j = 0; j < 200; ++j) {
                matrix(i, j) = dis(gen);
            }
        }
    }

    void run() override {
        result = matrix * scalar;
        // Prevent optimization
        volatile double dummy = result(0, 0);
    }

    void tearDown() override {
        // Cleanup
    }

private:
    Matrix matrix;
    Matrix result;
    double scalar;
};

// Benchmark: Matrix transform (reshape)
class MatrixTransformBenchmark : public perfbench::Benchmark {
public:
    MatrixTransformBenchmark() : Benchmark("MatrixTransform") {
        setDescription("Matrix transform (reshape) from 100x100 to 50x200");
    }

    void setUp() override {
        // Create a 100x100 matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        matrix = Matrix(100, 100);

        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < 100; ++j) {
                matrix(i, j) = dis(gen);
            }
        }
    }

    void run() override {
        result = matrix.transform(50, 200);
        // Prevent optimization
        volatile double dummy = result(0, 0);
    }

    void tearDown() override {
        // Cleanup
    }

private:
    Matrix matrix;
    Matrix result;
};

// Manual registration
static const perfbench::BenchmarkRegistrar<MatrixMultiplicationBenchmark> matrix_mult_registrar;
static const perfbench::BenchmarkRegistrar<MatrixOptimizedMultiplicationBenchmark> matrix_opt_mult_registrar;
static const perfbench::BenchmarkRegistrar<MatrixAdditionBenchmark> matrix_add_registrar;
static const perfbench::BenchmarkRegistrar<MatrixTransposeBenchmark> matrix_transpose_registrar;
static const perfbench::BenchmarkRegistrar<MatrixScalarMultiplicationBenchmark> matrix_scalar_mult_registrar;
static const perfbench::BenchmarkRegistrar<MatrixTransformBenchmark> matrix_transform_registrar;
