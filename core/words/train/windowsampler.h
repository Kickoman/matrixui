#pragma once

#include "core/lib/random.h"
#include "core/words/data/corpus.h"
#include "core/words/train/subsampler.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace Words {

class WindowSampler {
public:
    explicit WindowSampler(std::size_t window = 5) : window(window) {}

    template<typename Fn>
    void forEachPair(const TCorpus& chunk, XorShift& rng, Fn&& emit) const {
        const auto size = chunk.size();
        for (std::size_t i = 0; i < size; ++i) {
            const auto radius = rng.nextInteger(window) + 1;
            const std::size_t from = (i >= radius) ? i - radius : 0;
            const auto to = std::min<std::size_t>(size - 1, i + radius);

            for (std::size_t j = from; j <= to; ++j) {
                if (i == j) {
                    continue;
                }
                emit(Pair{chunk[i], chunk[j]});
            }
        }
    }

    std::size_t getWindow() const { return window; }

    double getPairsPerToken() const { return 1. * window + 1; }

private:
    std::size_t window;
};

template<typename Fn, typename Predicate>
void GeneratePairsWhile(
    const TCorpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    XorShift& rng,
    Fn&& emit,
    Predicate&& keepGoing,
    const std::size_t chunkSize = 1000,
    const std::size_t from = 0,
    std::size_t to = std::numeric_limits<std::size_t>::max()
) {
    to = std::min(corpus.size(), to);

    TCorpus chunk;
    chunk.reserve(chunkSize);

    for (std::size_t start = from; start < to; start += chunkSize) {
        if (!keepGoing()) {
            return;
        }

        const std::size_t finish = std::min(start + chunkSize, to);
        chunk.clear();
        for (std::size_t i = start; i < finish; ++i) {
            if (subsampler.shouldKeep(corpus[i], rng)) {
                chunk.push_back(corpus[i]);
            }
        }

        windowSampler.forEachPair(chunk, rng, emit);
    }
}

template<typename Fn>
void GeneratePairs(
    const TCorpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    XorShift& rng,
    Fn&& emit,
    const std::size_t chunkSize = 1000,
    const std::size_t from = 0,
    const std::size_t to = std::numeric_limits<std::size_t>::max()
) {
    GeneratePairsWhile(
        corpus, subsampler, windowSampler, rng, std::forward<Fn>(emit),
        [] { return true; }, chunkSize, from, to);
}

}
