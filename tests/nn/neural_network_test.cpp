#include <doctest/doctest.h>

#include "core/nn/neural_network.h"

#include <string>
#include <vector>

using Neural::LayersToTextRepresentation;
using Neural::TextRepresentationToLayers;

TEST_CASE("LayersToTextRepresentation joins the sizes with ', '") {
    CHECK(LayersToTextRepresentation({}) == "");
    CHECK(LayersToTextRepresentation({16}) == "16");
    CHECK(LayersToTextRepresentation({16, 32, 8}) == "16, 32, 8");
}

TEST_CASE("TextRepresentationToLayers reads a comma separated list") {
    const auto layers = TextRepresentationToLayers("16,32,8");
    REQUIRE(layers.size() == 3);
    CHECK(layers[0] == 16);
    CHECK(layers[1] == 32);
    CHECK(layers[2] == 8);
}

TEST_CASE("TextRepresentationToLayers ignores padding around the sizes") {
    const auto layers = TextRepresentationToLayers("  16 ,\t32\t, 8  ");
    REQUIRE(layers.size() == 3);
    CHECK(layers[0] == 16);
    CHECK(layers[2] == 8);
}

TEST_CASE("TextRepresentationToLayers drops whatever is not a size") {
    // The dialog parses as the user types, so a half-written list must not
    // throw -- anything that is not exactly one number is simply left out.
    CHECK(TextRepresentationToLayers("").empty());
    CHECK(TextRepresentationToLayers("   ").empty());
    CHECK(TextRepresentationToLayers("abc").empty());
    CHECK(TextRepresentationToLayers("-3").empty());
    CHECK(TextRepresentationToLayers("16abc").empty());
    CHECK(TextRepresentationToLayers("1.5").empty());

    const auto layers = TextRepresentationToLayers("16,,abc, 8 ,");
    REQUIRE(layers.size() == 2);
    CHECK(layers[0] == 16);
    CHECK(layers[1] == 8);
}

TEST_CASE("TextRepresentationToLayers accepts an explicit plus sign") {
    // Shared with the CSV parser, which has always taken "+1" -- std::from_chars
    // on its own would not.
    const auto layers = TextRepresentationToLayers("+16, 8");
    REQUIRE(layers.size() == 2);
    CHECK(layers[0] == 16);
    CHECK(layers[1] == 8);
}

TEST_CASE("TextRepresentationToLayers round-trips what LayersToTextRepresentation writes") {
    const std::vector<std::size_t> layers{784, 128, 10};
    CHECK(TextRepresentationToLayers(LayersToTextRepresentation(layers)) == layers);
}
