#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/train/subsampler.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"

#include <cmath>

using namespace Words;

TEST_CASE("A disabled subsampler keeps everything") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const Subsampler subsampler(vocabulary, 0.);

    XorShift rng(1);
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        CHECK(subsampler.getKeepProbability(id) == 1.f);
        CHECK(subsampler.shouldKeep(id, rng));
    }
    CHECK(subsampler.getAffectedWordsCount() == 0);

    // Must report the whole corpus, not zero: this figure drives
    // EstimateTotalPairs and therefore the learning-rate schedule.
    CHECK(subsampler.getExpectedCorpusLength(vocabulary)
          == doctest::Approx(static_cast<double>(vocabulary.getKeptTokens())));
}

TEST_CASE("Keep probability is non-decreasing as words get rarer") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const Subsampler subsampler(vocabulary, 1e-2);

    // Ids are ordered most- to least-frequent, so keep probability must rise.
    for (TWordId id = 1; id < vocabulary.getSize(); ++id) {
        CHECK(subsampler.getKeepProbability(id) >= subsampler.getKeepProbability(id - 1));
    }
}

TEST_CASE("Keep probability follows sqrt(t/f) + t/f, capped at one") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    constexpr double sample = 1e-2;
    const Subsampler subsampler(vocabulary, sample);

    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const double ratio = sample / vocabulary.getFrequency(id);
        const double expected = std::min(1., std::sqrt(ratio) + ratio);
        CHECK(subsampler.getKeepProbability(id) == doctest::Approx(expected).epsilon(1e-6));
    }
}

TEST_CASE("Only words above the threshold are affected") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const Subsampler subsampler(vocabulary, 1e-2);

    std::size_t affected = 0;
    for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
        if (subsampler.getKeepProbability(id) < 1.f) {
            ++affected;
        }
    }
    CHECK(subsampler.getAffectedWordsCount() == affected);
    // "a" holds half the corpus, so it must be downsampled.
    CHECK(subsampler.getKeepProbability(0) < 1.f);
}

TEST_CASE("shouldKeep is deterministic for a fixed seed") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const Subsampler subsampler(vocabulary, 1e-3);

    const auto sample = [&] {
        XorShift rng(4242);
        std::vector<bool> decisions;
        for (int i = 0; i < 500; ++i) {
            decisions.push_back(subsampler.shouldKeep(i % vocabulary.getSize(), rng));
        }
        return decisions;
    };

    CHECK(sample() == sample());
}

TEST_CASE("Expected corpus length predicts the observed survival rate") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const Subsampler subsampler(vocabulary, 1e-2);

    const double expected = subsampler.getExpectedCorpusLength(vocabulary);
    CHECK(expected > 0.);
    CHECK(expected <= static_cast<double>(vocabulary.getKeptTokens()));

    // Draw the whole corpus many times and compare the survival rate.
    XorShift rng(7);
    std::size_t kept = 0;
    constexpr int rounds = 400;
    for (int round = 0; round < rounds; ++round) {
        for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
            for (std::size_t n = 0; n < vocabulary.getCount(id); ++n) {
                if (subsampler.shouldKeep(id, rng)) {
                    ++kept;
                }
            }
        }
    }
    const double observed = static_cast<double>(kept) / rounds;
    CHECK(observed == doctest::Approx(expected).epsilon(0.05));
}
