#include "core/words/diagnostics/diagnostics.h"

#include "core/lib/random.h"
#include "core/words/negativesampler.h"
#include "core/words/subsampler.h"
#include "core/words/vocabulary.h"
#include "core/words/windowsampler.h"

#include <algorithm>
#include <functional>
#include <limits>

namespace Words::Diagnostics {

SubsamplerReport CheckSubsampler(
    const Vocabulary& vocabulary,
    const TCorpus& corpus,
    const double sample,
    const std::size_t topWords,
    const std::size_t previewTokens
) {
    const Subsampler subsampler(vocabulary, sample);

    SubsamplerReport report;
    report.sample = sample;
    report.vocabularySize = vocabulary.getSize();
    report.affectedWords = subsampler.getAffectedWordsCount();
    report.corpusSize = corpus.size();
    report.expectedCorpusLength = subsampler.getExpectedCorpusLength(vocabulary);

    const auto shown = std::min<std::size_t>(topWords, vocabulary.getSize());
    report.topKeepProbabilities.reserve(shown);
    for (TWordId id = 0; id < shown; ++id) {
        report.topKeepProbabilities.push_back({
            id,
            vocabulary.getFrequency(id),
            subsampler.getKeepProbability(id),
        });
    }

    XorShift rng(12345);
    const auto kept = Subsample(corpus, 0, corpus.size(), subsampler, rng);
    report.actualCorpusLength = kept.size();

    const auto beforeCount = std::min(previewTokens, corpus.size());
    report.before.assign(corpus.begin(), corpus.begin() + beforeCount);

    const auto afterCount = std::min(previewTokens, kept.size());
    report.after.assign(kept.begin(), kept.begin() + afterCount);

    return report;
}

WindowSamplerReport CheckWindowSampler(
    const Vocabulary& vocabulary,
    const TCorpus& corpus,
    const std::size_t window,
    const double sample
) {
    const Subsampler subsampler(vocabulary, sample);
    const WindowSampler windowSampler(window);

    WindowSamplerReport report;

    // --- a short chunk, printed verbatim so the pairing is easy to eyeball ---
    {
        const auto size = std::min<std::size_t>(8, corpus.size());
        report.toyChunk.assign(corpus.begin(), corpus.begin() + size);

        XorShift rng(42);
        windowSampler.forEachPair(report.toyChunk, rng, [&](const Pair& pair) {
            report.toyPairs.push_back(pair);
        });
    }

    // --- invariants on a chunk of distinct tokens ---------------------------
    //
    // Distinct ids matter: forEachPair only skips the centre by index, so a
    // repeated word would legitimately show up as its own context.
    {
        constexpr std::size_t uniqueTokens = 20;
        TCorpus unique(uniqueTokens);
        for (std::size_t i = 0; i < uniqueTokens; ++i) {
            unique[i] = static_cast<TWordId>(i);
        }

        XorShift rng(7);
        bool selfPair = false;
        std::vector<std::size_t> emitted(uniqueTokens, 0);
        windowSampler.forEachPair(unique, rng, [&](const Pair& pair) {
            if (pair.center == pair.context) {
                selfPair = true;
            }
            ++emitted[pair.center];
        });

        report.selfPair = {"self-pair check", !selfPair};
        report.coverage = {
            "coverage check",
            std::find(emitted.begin(), emitted.end(), 0u) == emitted.end(),
        };
    }

    // --- a full pass over the corpus ---------------------------------------
    {
        XorShift rng(2024);
        std::vector<std::size_t> asCenter(vocabulary.getSize(), 0);

        GeneratePairs(corpus, subsampler, windowSampler, rng, [&](const Pair& pair) {
            ++report.pairsGenerated;
            ++asCenter[pair.center];
        });

        report.expectedUpperBound = static_cast<std::size_t>(
            subsampler.getExpectedCorpusLength(vocabulary) * windowSampler.getPairsPerToken());

        std::vector<std::pair<std::size_t, TWordId>> byCount;
        byCount.reserve(vocabulary.getSize());
        for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
            byCount.emplace_back(asCenter[id], id);
        }

        const auto shown = std::min<std::size_t>(10, byCount.size());
        std::partial_sort(byCount.begin(), byCount.begin() + shown, byCount.end(), std::greater<>());
        report.topCenters.assign(byCount.begin(), byCount.begin() + shown);
    }

    return report;
}

}  // namespace Words::Diagnostics
