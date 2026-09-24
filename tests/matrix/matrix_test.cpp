#include <doctest/doctest.h>

#include "core/matrix/matrix.h"

#include <stdexcept>
#include <vector>

namespace {

Matrix Of(const std::vector<std::vector<double>>& rows) {
    return Matrix(rows);
}

}  // namespace

TEST_CASE("multiplyAdd adds the bias row to every row of the product") {
    const Matrix input = Of({{1., 2., 3.},
                             {4., 5., 6.}});
    const Matrix weights = Of({{1., 0.},
                               {0., 1.},
                               {1., 1.}});
    const Matrix bias = Of({{10., 20.}});

    const Matrix result = input.multiplyAdd(weights, bias);

    REQUIRE(result.getRows() == 2);
    REQUIRE(result.getCols() == 2);
    CHECK(result(0, 0) == doctest::Approx(14.));
    CHECK(result(0, 1) == doctest::Approx(25.));
    CHECK(result(1, 0) == doctest::Approx(20.));
    CHECK(result(1, 1) == doctest::Approx(31.));
}

TEST_CASE("multiplyAdd gives every identical row the same answer") {
    const Matrix weights = Of({{0.5, -1.5},
                               {2.0,  0.25},
                               {-0.75, 3.0}});
    const Matrix bias = Of({{0.125, -0.5}});
    const std::vector<double> row{1.5, -2.25, 0.75};

    const Matrix single = Of({row}).multiplyAdd(weights, bias);
    const Matrix batched = Of({row, row, row, row}).multiplyAdd(weights, bias);

    REQUIRE(batched.getRows() == 4);
    REQUIRE(batched.getCols() == single.getCols());
    for (std::size_t r = 0; r < batched.getRows(); ++r) {
        for (std::size_t c = 0; c < batched.getCols(); ++c) {
            CHECK(batched(r, c) == doctest::Approx(single(0, c)).epsilon(1e-14));
        }
    }
}

TEST_CASE("multiplyAdd survives a product past Eigen's blocking threshold") {
    // This asserts almost nothing on purpose. Its job is to run a product big
    // enough that Eigen takes its blocked, vectorised path, because that is where a
    // build whose EIGEN_MAX_ALIGN_BYTES disagrees with its enabled kernels dies:
    // the vendored Eigen 3.3.90 turns on AVX-512 kernels on a -march that has them
    // while still promising 32-byte alignment, and the kernel then issues a
    // 64-byte-aligned store. The root CMakeLists pins the value to 64 to keep the
    // two in step; if that is ever lost, this case segfaults and every other test
    // here keeps passing, because every other matrix in the suite is below the
    // threshold. That is exactly how the problem stayed hidden.
    const std::size_t rows = 256;
    const std::size_t inputs = 784;
    const std::size_t outputs = 128;

    const Matrix input(rows, inputs, 0.5);
    const Matrix weights(inputs, outputs, 0.5);
    const Matrix bias(1, outputs, 1.5);

    const Matrix result = input.multiplyAdd(weights, bias);

    REQUIRE(result.getRows() == rows);
    REQUIRE(result.getCols() == outputs);
    const double expected = static_cast<double>(inputs) * 0.25 + 1.5;
    CHECK(result(0, 0) == doctest::Approx(expected));
    CHECK(result(rows - 1, outputs - 1) == doctest::Approx(expected));
}

TEST_CASE("multiplyAdd refuses a multiplier the left side cannot meet") {
    const Matrix input = Of({{1., 2., 3.}});
    const Matrix bias = Of({{0., 0.}});

    SUBCASE("one row too many") {
        const Matrix weights(4, 2, 1.);
        CHECK_THROWS_AS(input.multiplyAdd(weights, bias), std::invalid_argument);
    }

    SUBCASE("one row too few") {
        const Matrix weights(2, 2, 1.);
        CHECK_THROWS_AS(input.multiplyAdd(weights, bias), std::invalid_argument);
    }
}
