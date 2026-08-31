#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/negativesampler.h"
#include "core/words/vocabulary.h"
#include "tests/support/fixtures.h"

#include <cmath>
#include <unordered_set>

using namespace Words;

namespace {

// Small enough to build quickly, large enough for the distribution to settle.
constexpr std::size_t kTableSize = 200'000;

}  // namespace

TEST_CASE("NegativeSampler rejects impossible configurations") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    CHECK_THROWS_AS(NegativeSampler(vocabulary, /*tableSize=*/2), std::runtime_error);
    CHECK(NegativeSampler(vocabulary, kTableSize).getTableSize() == kTableSize);
}

TEST_CASE("Every word in the vocabulary is reachable") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const NegativeSampler sampler(vocabulary, kTableSize);

    std::unordered_set<TWordId> seen;
    XorShift rng(1);
    for (std::size_t i = 0; i < 200'000; ++i) {
        seen.insert(sampler.sample(rng));
    }
    CHECK(seen.size() == vocabulary.getSize());
}

TEST_CASE("Sampling frequency follows the count^0.75 distribution") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    constexpr double power = 0.75;
    const NegativeSampler sampler(vocabulary, kTableSize, power);

    // Expected probabilities.
    double total = 0.;
    std::vector<double> weights(vocabulary.getSize());
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        weights[id] = std::pow(static_cast<double>(vocabulary.getCount(id)), power);
        total += weights[id];
    }

    constexpr std::size_t draws = 400'000;
    std::vector<std::size_t> hits(vocabulary.getSize(), 0);
    XorShift rng(2);
    for (std::size_t i = 0; i < draws; ++i) {
        ++hits[sampler.sample(rng)];
    }

    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const double expected = weights[id] / total;
        const double observed = static_cast<double>(hits[id]) / draws;
        // The table is built by walking cumulative mass, so it is a coarse
        // quantisation of the target distribution rather than an exact one.
        CHECK(observed == doctest::Approx(expected).epsilon(0.05));
    }
}

TEST_CASE("The ^0.75 exponent flattens the frequency distribution") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    const auto top = vocabulary.getCount(0);
    const auto rare = vocabulary.getCount(vocabulary.getSize() - 1);
    const double rawRatio = static_cast<double>(top) / static_cast<double>(rare);
    const double flatRatio = std::pow(static_cast<double>(top), 0.75)
                           / std::pow(static_cast<double>(rare), 0.75);

    CHECK(flatRatio < rawRatio);
    CHECK(flatRatio == doctest::Approx(std::pow(rawRatio, 0.75)));
}

TEST_CASE("sampleExcluding almost never returns the excluded word") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const NegativeSampler sampler(vocabulary, kTableSize);

    // Note: exclusion is best-effort. After 8 failed attempts the
    // implementation falls back to an unfiltered sample, so a leak is possible
    // in principle -- this pins the rate as negligible rather than zero.
    constexpr TWordId excluded = 0;  // the most frequent word, the worst case
    constexpr std::size_t draws = 200'000;

    XorShift rng(3);
    std::size_t leaks = 0;
    for (std::size_t i = 0; i < draws; ++i) {
        if (sampler.sampleExcluding(excluded, rng) == excluded) {
            ++leaks;
        }
    }
    CHECK(leaks * 1000 < draws);  // well under 0.1%
}

TEST_CASE("getProbability agrees with the observed sampling rate") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);
    const NegativeSampler sampler(vocabulary, kTableSize);

    constexpr std::size_t draws = 200'000;
    XorShift rng(4);
    std::vector<std::size_t> hits(vocabulary.getSize(), 0);
    for (std::size_t i = 0; i < draws; ++i) {
        ++hits[sampler.sample(rng)];
    }

    double totalProbability = 0.;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const double declared = sampler.getProbability(id);
        totalProbability += declared;
        CHECK(static_cast<double>(hits[id]) / draws == doctest::Approx(declared).epsilon(0.05));
    }
    CHECK(totalProbability == doctest::Approx(1.).epsilon(1e-9));
}
