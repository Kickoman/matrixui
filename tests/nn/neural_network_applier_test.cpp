#include <doctest/doctest.h>

#include "core/matrix/matrix.h"
#include "core/nn/neural_network.h"
#include "core/nn/neural_network_applier.h"
#include "core/nn/neural_network_loader.h"

#include <stdexcept>
#include <vector>

namespace {

Neural::NeuralNetwork Network(std::vector<std::size_t> layers) {
    Neural::NeuralNetworkConfiguration configuration{};
    configuration.hiddenActivation = Neural::ActivationType::ReLU;
    configuration.outputActivation = Neural::ActivationType::Softmax;
    configuration.layersSizes = std::move(layers);
    return Neural::CreateNetwork(configuration);
}

std::vector<double> Row(const std::size_t width) {
    std::vector<double> values(width);
    for (std::size_t i = 0; i < width; ++i) {
        values[i] = static_cast<double>((i * 37) % 251) / 251.;
    }
    return values;
}

Matrix Repeated(const std::vector<double>& row, const std::size_t rows) {
    Matrix input(rows, row.size());
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < row.size(); ++c) {
            input(r, c) = row[c];
        }
    }
    return input;
}

}  // namespace

TEST_CASE("Predict on identical rows answers each of them with the single-row answer") {
    const Neural::NeuralNetwork network = Network({8, 6, 4});
    const std::vector<double> row = Row(8);

    const Matrix single = Neural::Predict(network, Repeated(row, 1));
    const Matrix batched = Neural::Predict(network, Repeated(row, 4));

    REQUIRE(single.getRows() == 1);
    REQUIRE(batched.getRows() == 4);
    REQUIRE(batched.getCols() == single.getCols());
    for (std::size_t r = 0; r < batched.getRows(); ++r) {
        for (std::size_t c = 0; c < batched.getCols(); ++c) {
            CHECK(batched(r, c) == doctest::Approx(single(0, c)).epsilon(1e-14));
        }
    }
}

TEST_CASE("Predict keeps every row of a batch a distribution") {
    const Neural::NeuralNetwork network = Network({8, 6, 4});
    const Matrix batched = Neural::Predict(network, Repeated(Row(8), 5));

    for (std::size_t r = 0; r < batched.getRows(); ++r) {
        double sum = 0.;
        for (std::size_t c = 0; c < batched.getCols(); ++c) {
            sum += batched(r, c);
        }
        CHECK(sum == doctest::Approx(1.).epsilon(1e-12));
    }
}

TEST_CASE("Predict refuses a row that is not the network's width") {
    // A wrong width reaches raw Eigen, whose own check is an eigen_assert that
    // -DNDEBUG removes: one column too many reads past the weights, one too few
    // silently truncates the dot product. Both must be refused here.
    const Neural::NeuralNetwork network = Network({8, 6, 4});

    SUBCASE("one column too many") {
        CHECK_THROWS_AS(Neural::Predict(network, Repeated(Row(9), 1)), std::invalid_argument);
    }

    SUBCASE("one column too few") {
        CHECK_THROWS_AS(Neural::Predict(network, Repeated(Row(7), 1)), std::invalid_argument);
    }

    SUBCASE("far too few") {
        CHECK_THROWS_AS(Neural::Predict(network, Repeated(Row(1), 1)), std::invalid_argument);
    }

    SUBCASE("a batch of the wrong width") {
        CHECK_THROWS_AS(Neural::Predict(network, Repeated(Row(9), 4)), std::invalid_argument);
    }
}

TEST_CASE("Predict refuses an overlong row on a topology where the overread escapes") {
    // Pinned to 64-16-10 on purpose, and this case earns its keep only under
    // tests/serving/run_sanitizers.sh. Whether a one-column overread leaves the
    // weights allocation is not monotone in either dimension: 8x10 weights hide
    // it, 16x5 weights do not. On a toy 8-6-4 the read stays inside the block
    // Eigen already over-allocated and the sanitizer says nothing, so a guard
    // removed here would look clean.
    const Neural::NeuralNetwork network = Network({64, 16, 10});
    CHECK_THROWS_AS(Neural::Predict(network, Repeated(Row(65), 1)), std::invalid_argument);
    CHECK_THROWS_AS(Neural::Predict(network, Repeated(Row(65), 4)), std::invalid_argument);
}

TEST_CASE("Predict accepts the width the network declares") {
    const Neural::NeuralNetwork network = Network({8, 6, 4});
    CHECK_NOTHROW(Neural::Predict(network, Repeated(Row(8), 1)));
    CHECK_NOTHROW(Neural::Predict(network, Repeated(Row(8), 3)));
}
