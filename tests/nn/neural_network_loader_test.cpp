#include <doctest/doctest.h>

#include "core/lib/write.h"
#include "core/nn/layers.h"
#include "core/nn/neural_network.h"
#include "core/nn/neural_network_loader.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

using Neural::ActivationType;
using Neural::DenseLayer;
using Neural::NeuralNetwork;
using Neural::NeuralNetworkConfiguration;

NeuralNetworkConfiguration Config(
    std::vector<std::size_t> layers,
    const ActivationType hidden = ActivationType::ReLU,
    const ActivationType output = ActivationType::Softmax
) {
    NeuralNetworkConfiguration configuration{};
    configuration.hiddenActivation = hidden;
    configuration.outputActivation = output;
    configuration.layersSizes = std::move(layers);
    return configuration;
}

std::string Serialized(const NeuralNetwork& network) {
    std::ostringstream out(std::ios::binary);
    Neural::SaveNetwork(out, network);
    return out.str();
}

NeuralNetwork Deserialized(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    return Neural::LoadNetwork(in);
}

// version(4) + activations(2) + numLayers(4) + one uint64 per layer size.
std::size_t HeaderBytes(const std::size_t numLayers) {
    return 10 + numLayers * sizeof(std::uint64_t);
}

// A header with hand-chosen fields, for the malformed cases SaveNetwork cannot
// produce. numLayers is written separately from `sizes` so the two can disagree.
std::string RawHeader(
    const std::uint32_t version,
    const std::uint32_t numLayers,
    const std::vector<std::uint64_t>& sizes
) {
    std::ostringstream out(std::ios::binary);
    WriteBinaryLE(out, version);
    WriteBinaryLE(out, static_cast<std::uint8_t>(ActivationType::ReLU));
    WriteBinaryLE(out, static_cast<std::uint8_t>(ActivationType::Softmax));
    WriteBinaryLE(out, numLayers);
    for (const auto size : sizes) {
        WriteBinaryLE(out, size);
    }
    return out.str();
}

// One letter per layer variant, so a whole stack can be compared as a string.
std::string StackShape(const NeuralNetwork& network) {
    std::string shape;
    for (const auto& layer : network.layerStack) {
        shape += std::visit([](const auto& value) -> char {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DenseLayer>) {
                return 'D';
            } else if constexpr (std::is_same_v<T, Neural::ActivationLayer>) {
                return 'A';
            } else if constexpr (std::is_same_v<T, Neural::DropoutLayer>) {
                return 'O';
            } else {
                return 'S';
            }
        }, layer);
    }
    return shape;
}

std::vector<const DenseLayer*> DenseLayers(const NeuralNetwork& network) {
    std::vector<const DenseLayer*> dense;
    for (const auto& layer : network.layerStack) {
        if (const auto* value = std::get_if<DenseLayer>(&layer)) {
            dense.push_back(value);
        }
    }
    return dense;
}

}  // namespace


TEST_CASE("LoadNetwork restores the configuration SaveNetwork wrote") {
    const auto original = Neural::CreateNetwork(
        Config({4, 3, 2}, ActivationType::Tanh, ActivationType::Sigmoid));
    const auto restored = Deserialized(Serialized(original));

    CHECK(restored.config.layersSizes == original.config.layersSizes);
    CHECK(restored.hiddenActivation() == ActivationType::Tanh);
    CHECK(restored.outputActivation() == ActivationType::Sigmoid);
    CHECK(restored.inputSize() == 4);
    CHECK(restored.outputSize() == 2);
    CHECK_FALSE(restored.empty());
}

TEST_CASE("LoadNetwork restores the weights bit-exactly") {
    const auto original = Neural::CreateNetwork(Config({4, 3, 2}));
    const auto restored = Deserialized(Serialized(original));

    const auto before = DenseLayers(original);
    const auto after = DenseLayers(restored);
    REQUIRE(after.size() == before.size());

    for (std::size_t layer = 0; layer < before.size(); ++layer) {
        const auto& source = before[layer]->weights;
        const auto& target = after[layer]->weights;
        REQUIRE(target.getRows() == source.getRows());
        REQUIRE(target.getCols() == source.getCols());
        for (std::size_t row = 0; row < source.getRows(); ++row) {
            for (std::size_t col = 0; col < source.getCols(); ++col) {
                REQUIRE(target(row, col) == source(row, col));
            }
        }
        for (std::size_t col = 0; col < source.getCols(); ++col) {
            REQUIRE(after[layer]->biases(0, col) == before[layer]->biases(0, col));
        }
    }
}

TEST_CASE("LoadNetwork rebuilds the stack CreateNetwork assembles") {
    // D dense, A activation, O dropout, S softmax. A Softmax output collapses
    // into a SoftmaxLayer; any other output activation stays an ActivationLayer.
    // Nothing else pins this order, and the serving loader depends on it.
    const auto shapeOf = [](const NeuralNetworkConfiguration& configuration) {
        const auto created = Neural::CreateNetwork(configuration);
        const auto restored = Deserialized(Serialized(created));
        CHECK(StackShape(restored) == StackShape(created));
        return StackShape(restored);
    };

    CHECK(shapeOf(Config({4, 3, 2})) == "DAODS");
    CHECK(shapeOf(Config({4, 3, 2}, ActivationType::ReLU, ActivationType::Sigmoid)) == "DAODA");
    CHECK(shapeOf(Config({4, 2})) == "DS");
    CHECK(shapeOf(Config({4, 3, 3, 2})) == "DAODAODS");
}

TEST_CASE("LoadNetwork refuses a version it does not know") {
    // Pinned by the classifier and generator golden snapshots, which feed the
    // literal "garbage": its first four bytes read as 1651663207.
    try {
        Deserialized("garbage");
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()) == "Unsupported network version: 1651663207");
    }

    CHECK_THROWS_AS(Deserialized(RawHeader(2, 2, {4, 2})), std::runtime_error);
}


// --- A damaged file is refused, and the refusal names numbers ---------------
//
// Every case below used to load silently or walk off the end of a vector.

TEST_CASE("A file cut off after the header is refused") {
    const auto original = Neural::CreateNetwork(Config({4, 3, 2}));
    const auto truncated = Serialized(original).substr(0, HeaderBytes(3));

    // 4x3 weights + 3 biases + 3x2 weights + 2 biases, eight bytes each.
    try {
        Deserialized(truncated);
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network file is shorter than its header claims "
                 "(expected 184 bytes of weights, found 0)");
    }
}

TEST_CASE("A file cut off mid-weights is refused") {
    const auto original = Neural::CreateNetwork(Config({4, 3, 2}));
    const std::string complete = Serialized(original);
    const std::size_t firstTransition = (4 * 3 + 3) * sizeof(double);

    try {
        Deserialized(complete.substr(0, HeaderBytes(3) + firstTransition));
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network file is shorter than its header claims "
                 "(expected 184 bytes of weights, found 120)");
    }

    // One byte short of complete still counts as short.
    CHECK_THROWS_AS(Deserialized(complete.substr(0, complete.size() - 1)), std::runtime_error);
    CHECK_NOTHROW(Deserialized(complete));
}

TEST_CASE("An empty file is refused instead of reading an indeterminate version") {
    try {
        Deserialized("");
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()) == "Network file is truncated: no version header");
    }

    CHECK_THROWS_AS(Deserialized(std::string("\x01\x00\x00", 3)), std::runtime_error);
}

TEST_CASE("A header that declares too few layers is refused") {
    try {
        Deserialized(RawHeader(1, 0, {}));
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network file declares 0 layers, at least 2 are required");
    }

    CHECK_THROWS_AS(Deserialized(RawHeader(1, 1, {4})), std::runtime_error);
}

TEST_CASE("A header that declares four billion layers is refused before allocating") {
    try {
        Deserialized(RawHeader(1, 0xFFFFFFFF, {}));
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network file declares 4294967295 layers, more than the 4096 allowed");
    }
}

TEST_CASE("A layer count the file cannot back is refused") {
    try {
        Deserialized(RawHeader(1, 64, {4, 2}));
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network file declares 64 layers but holds only 16 bytes after the header");
    }
}

TEST_CASE("A layer size outside the allowed range is refused") {
    // 2^63 * 2 wraps to zero in uint64, so without this bound the expected
    // weight volume would come out as 16 bytes and a short file would pass.
    try {
        Deserialized(RawHeader(1, 2, {std::uint64_t{1} << 63, 2}));
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network layer 0 declares an unusable size: 9223372036854775808 "
                 "(allowed 1..16777216)");
    }

    CHECK_THROWS_AS(Deserialized(RawHeader(1, 2, {4, 0})), std::runtime_error);
}

TEST_CASE("A weight volume over the ceiling is refused") {
    // Each size clears the per-layer bound; only the product does not.
    try {
        Deserialized(RawHeader(1, 2, {std::uint64_t{1} << 24, std::uint64_t{1} << 24}));
        FAIL("expected a throw");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what())
              == "Network file declares more than 4294967296 bytes of weights");
    }
}
