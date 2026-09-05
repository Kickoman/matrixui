#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/data/embeddings.h"
#include <sstream>

#include <cmath>

using namespace Words;

TEST_CASE("dot computes the inner product") {
    const std::vector<TFloat> a{1.f, 2.f, 3.f};
    const std::vector<TFloat> b{4.f, -5.f, 6.f};
    // 4 - 10 + 18
    CHECK(dot(a.data(), b.data(), 3) == doctest::Approx(12.));
}

TEST_CASE("addScaled accumulates a scaled vector in place") {
    std::vector<TFloat> a{1.f, 2.f, 3.f};
    const std::vector<TFloat> b{10.f, 10.f, 10.f};

    addScaled(a.data(), b.data(), 0.5, 3);

    CHECK(a[0] == doctest::Approx(6.f));
    CHECK(a[1] == doctest::Approx(7.f));
    CHECK(a[2] == doctest::Approx(8.f));
}

TEST_CASE("initializeZero zeroes every element") {
    Embeddings embeddings(4, 8);
    embeddings.initializeZero();

    for (TWordId id = 0; id < 4; ++id) {
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(embeddings.row(id)[i] == 0.f);
        }
    }
}

TEST_CASE("initializeUniform respects the 0.5/dim bound and varies per row") {
    constexpr std::size_t words = 256;
    constexpr std::size_t dim = 16;
    const double bound = 0.5 / dim;

    Embeddings embeddings(words, dim);
    XorShift rng(1234);
    embeddings.initializeUniform(rng);

    double sum = 0.;
    double sumSquares = 0.;
    std::size_t count = 0;
    for (TWordId id = 0; id < words; ++id) {
        for (std::size_t i = 0; i < dim; ++i) {
            const double value = embeddings.row(id)[i];
            CHECK(std::abs(value) < bound);
            sum += value;
            sumSquares += value * value;
            ++count;
        }
    }
    // Centred on zero.
    const double mean = sum / count;
    CHECK(mean == doctest::Approx(0.).epsilon(0.05).scale(bound));

    // Standard deviation of a uniform(-b, b) distribution is b / sqrt(3).
    const double stddev = std::sqrt(sumSquares / count - mean * mean);
    CHECK(stddev == doctest::Approx(bound / std::sqrt(3.)).epsilon(0.05));

    // Distinct rows.
    CHECK(!std::equal(embeddings.row(0), embeddings.row(0) + dim, embeddings.row(1)));
}

TEST_CASE("Embeddings report their shape and byte size") {
    const Embeddings embeddings(10, 4);
    CHECK(embeddings.getWords() == 10);
    CHECK(embeddings.getDim() == 4);
    CHECK(embeddings.getBytes() == 10 * 4 * sizeof(TFloat));
}

TEST_CASE("Embeddings survive a Save/Load round-trip") {
    Embeddings original(5, 3);
    original.initializeZero();
    for (TWordId id = 0; id < 5; ++id) {
        for (std::size_t i = 0; i < 3; ++i) {
            original.row(id)[i] = static_cast<TFloat>(id * 10 + i);
        }
    }

    std::stringstream stream;
    Embeddings::Save(stream, original);
    const auto loaded = Embeddings::Load(stream);

    REQUIRE(loaded.getWords() == 5);
    REQUIRE(loaded.getDim() == 3);
    for (TWordId id = 0; id < 5; ++id) {
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(loaded.row(id)[i] == original.row(id)[i]);
        }
    }
}
