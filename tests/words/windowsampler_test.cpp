#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/train/subsampler.h"
#include "core/words/data/vocabulary.h"
#include "core/words/train/windowsampler.h"
#include "tests/support/fixtures.h"

#include <set>

using namespace Words;

namespace {

// Distinct ids: forEachPair only skips the centre by index, so repeated words
// would legitimately produce centre == context.
TCorpus DistinctChunk(const std::size_t size) {
    TCorpus chunk(size);
    for (std::size_t i = 0; i < size; ++i) {
        chunk[i] = static_cast<TWordId>(i);
    }
    return chunk;
}

}  // namespace

TEST_CASE("No pair has a word as its own context") {
    const WindowSampler sampler(5);
    const auto chunk = DistinctChunk(50);

    XorShift rng(7);
    bool selfPair = false;
    sampler.forEachPair(chunk, rng, [&](const Pair& pair) {
        if (pair.center == pair.context) {
            selfPair = true;
        }
    });
    CHECK_FALSE(selfPair);
}

TEST_CASE("Every token appears as a centre at least once") {
    const WindowSampler sampler(5);
    const auto chunk = DistinctChunk(20);

    XorShift rng(7);
    std::vector<std::size_t> asCenter(chunk.size(), 0);
    sampler.forEachPair(chunk, rng, [&](const Pair& pair) { ++asCenter[pair.center]; });

    for (const auto count : asCenter) {
        CHECK(count > 0);
    }
}

TEST_CASE("Context never lies outside the maximum window radius") {
    constexpr std::size_t window = 4;
    const WindowSampler sampler(window);
    const auto chunk = DistinctChunk(40);

    XorShift rng(11);
    sampler.forEachPair(chunk, rng, [&](const Pair& pair) {
        // Ids equal positions in this fixture.
        const auto distance = pair.center > pair.context
            ? pair.center - pair.context
            : pair.context - pair.center;
        CHECK(distance >= 1);
        CHECK(distance <= window);
    });
}

TEST_CASE("A single-token chunk emits no pairs") {
    const WindowSampler sampler(5);
    XorShift rng(1);
    std::size_t pairs = 0;
    sampler.forEachPair(DistinctChunk(1), rng, [&](const Pair&) { ++pairs; });
    CHECK(pairs == 0);
}

TEST_CASE("Pair generation is deterministic for a fixed seed") {
    const WindowSampler sampler(5);
    const auto chunk = DistinctChunk(30);

    const auto collect = [&] {
        XorShift rng(2024);
        std::vector<std::pair<TWordId, TWordId>> pairs;
        sampler.forEachPair(chunk, rng, [&](const Pair& p) { pairs.emplace_back(p.center, p.context); });
        return pairs;
    };

    CHECK(collect() == collect());
}

TEST_CASE("getPairsPerToken bounds the average pair count") {
    constexpr std::size_t window = 5;
    const WindowSampler sampler(window);
    CHECK(sampler.getWindow() == window);
    CHECK(sampler.getPairsPerToken() == doctest::Approx(6.));

    // Away from the chunk edges each token emits 2*radius pairs, and radius is
    // uniform on [1, window], so the mean is window + 1.
    const auto chunk = DistinctChunk(4000);
    XorShift rng(5);
    std::size_t pairs = 0;
    sampler.forEachPair(chunk, rng, [&](const Pair&) { ++pairs; });

    const double perToken = static_cast<double>(pairs) / chunk.size();
    CHECK(perToken == doctest::Approx(sampler.getPairsPerToken()).epsilon(0.05));
}

TEST_CASE("GeneratePairs honours the [from, to) range") {
    const auto vocabulary = Tests::ToyVocabulary(1);
    const Subsampler subsampler(vocabulary, 0.);  // disabled: keep every token
    const WindowSampler windowSampler(2);

    const auto corpus = DistinctChunk(100);

    XorShift rng(3);
    std::set<TWordId> centers;
    GeneratePairs(corpus, subsampler, windowSampler, rng,
        [&](const Pair& p) { centers.insert(p.center); },
        /*chunkSize=*/10, /*from=*/20, /*to=*/40);

    REQUIRE_FALSE(centers.empty());
    CHECK(*centers.begin() >= 20);
    CHECK(*centers.rbegin() < 40);
}
