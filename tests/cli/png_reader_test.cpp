#include <doctest/doctest.h>

#include "cli/lib/png_reader.h"

#include "core/matrix/matrix.h"
#include "core/png/pngreader.h"

#include "tests/support/temp_dir.h"

#include <stdexcept>

namespace {

Matrix Gradient(const std::size_t side) {
    return Matrix(side, side, [side](std::size_t row, std::size_t col) {
        return static_cast<double>(row * side + col) / static_cast<double>(side * side);
    });
}

}  // namespace

TEST_CASE("MakeCachedPngReader round-trips an image written by toImage") {
    Tests::TempDir dir;
    const auto image = Gradient(4);
    const auto path = dir.file("gradient.png");
    PngUtils::toImage(image, path.string());

    const auto reader = CliLib::MakeCachedPngReader(4, 4);
    const Matrix loaded = reader(path);

    REQUIRE(loaded.getRows() == 1);
    REQUIRE(loaded.getCols() == 16);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            // One byte of quantization each way.
            CHECK(loaded(0, row * 4 + col) == doctest::Approx(image(row, col)).epsilon(2.0 / 255.0));
        }
    }
    CHECK(loaded(0, 0) == doctest::Approx(0.0));  // 0 = white survives the round trip
}

TEST_CASE("MakeCachedPngReader resizes to the requested dimensions") {
    Tests::TempDir dir;
    const auto path = dir.file("big.png");
    PngUtils::toImage(Gradient(8), path.string());

    const auto reader = CliLib::MakeCachedPngReader(4, 4);
    const Matrix loaded = reader(path);

    CHECK(loaded.getRows() == 1);
    CHECK(loaded.getCols() == 16);
}

TEST_CASE("MakeCachedPngReader serves repeated reads from its cache") {
    Tests::TempDir dir;
    const auto path = dir.file("image.png");
    PngUtils::toImage(Matrix(4, 4, 0.0), path.string());

    const auto reader = CliLib::MakeCachedPngReader(4, 4);
    const Matrix first = reader(path);

    // Overwrite the file on disk; a copy of the reader must still return the
    // cached pixels, proving the cache is shared across std::function copies.
    PngUtils::toImage(Matrix(4, 4, 1.0), path.string());
    const auto copy = reader;
    const Matrix second = copy(path);

    for (std::size_t i = 0; i < 16; ++i) {
        CHECK(second(0, i) == doctest::Approx(first(0, i)));
    }
}

TEST_CASE("MakeCachedPngReader propagates a failure to load") {
    Tests::TempDir dir;
    const auto reader = CliLib::MakeCachedPngReader(4, 4);
    CHECK_THROWS_AS(reader(dir.file("absent.png")), std::runtime_error);
}
