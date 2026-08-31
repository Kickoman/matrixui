#include "core/words/diagnostics/diagnostics.h"

#include "core/lib/random.h"
#include "core/words/negativesampler.h"
#include "core/words/vocabulary.h"

#include <cmath>
#include <unordered_set>
#include <vector>

namespace Words::Diagnostics {

namespace {

constexpr double kPower = 0.75;
constexpr double kMinExpectedHits = 50.;

}  // namespace

NegativeSamplerReport CheckNegativeSampler(
    const Vocabulary& vocabulary,
    const std::size_t coverageDraws,
    const std::size_t distributionDraws,
    const std::size_t exclusionDraws
) {
    const NegativeSampler sampler(vocabulary);

    NegativeSamplerReport report;
    report.tableSize = sampler.getTableSize();
    report.vocabularySize = vocabulary.getSize();

    // --- every word must be reachable --------------------------------------
    {
        std::unordered_set<TWordId> seen;
        XorShift rng(1);
        for (std::size_t i = 0; i < coverageDraws; ++i) {
            seen.insert(sampler.sample(rng));
        }
        report.reachableWords = seen.size();
        report.coverage = {"reachable words", seen.size() == vocabulary.getSize()};
    }

    // --- the draw distribution must follow count^0.75 -----------------------
    {
        std::vector<std::size_t> hits(vocabulary.getSize(), 0);
        XorShift rng(2);
        for (std::size_t i = 0; i < distributionDraws; ++i) {
            ++hits[sampler.sample(rng)];
        }

        double total = 0.;
        std::vector<double> weights(vocabulary.getSize());
        for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
            weights[id] = std::pow(static_cast<double>(vocabulary.getCount(id)), kPower);
            total += weights[id];
        }

        double chiSquare = 0.;
        std::size_t counted = 0;
        double maxAbsZ = 0.;
        TWordId worstId = 0;

        for (TWordId id = 0; id < vocabulary.getSize(); ++id) {
            const double p = weights[id] / total;
            const double expected = p * static_cast<double>(distributionDraws);
            if (expected < kMinExpectedHits) {
                continue;
            }

            const double sigma = std::sqrt(expected * (1. - p));
            const double z = (static_cast<double>(hits[id]) - expected) / sigma;

            chiSquare += z * z;
            ++counted;

            if (std::abs(z) > maxAbsZ) {
                maxAbsZ = std::abs(z);
                worstId = id;
            }
        }

        report.wordsTested = counted;
        if (counted > 0) {
            report.reducedChiSquare = chiSquare / static_cast<double>(counted);
            report.tolerance = 4. * std::sqrt(2. / static_cast<double>(counted));
            report.expectedMaxZ = std::sqrt(2. * std::log(2. * static_cast<double>(counted)));
        }
        report.maxAbsZ = maxAbsZ;
        report.worstId = worstId;
        report.distribution = {
            "chi2 / df",
            counted > 0 && std::abs(report.reducedChiSquare - 1.) < report.tolerance,
        };
    }

    // --- the ^0.75 exponent flattens the distribution -----------------------
    {
        const auto top = vocabulary.getCount(0);
        const auto rare = vocabulary.getCount(vocabulary.getSize() - 1);
        if (rare > 0) {
            report.rawCountRatio = static_cast<double>(top) / static_cast<double>(rare);
            report.flattenedRatio = std::pow(static_cast<double>(top), kPower)
                                  / std::pow(static_cast<double>(rare), kPower);
        }
    }

    // --- sampleExcluding must (almost) never return the excluded word -------
    {
        XorShift rng(3);
        constexpr TWordId excluded = 0;
        bool leaked = false;
        for (std::size_t i = 0; i < exclusionDraws; ++i) {
            if (sampler.sampleExcluding(excluded, rng) == excluded) {
                leaked = true;
                break;
            }
        }
        report.exclusion = {"exclusion check", !leaked};
    }

    return report;
}

}  // namespace Words::Diagnostics
