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
