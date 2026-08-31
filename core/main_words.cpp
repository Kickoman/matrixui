#include <iostream>

#include "core/lib/random.h"
#include "core/words/subsampler.h"
#include "core/words/vocabulary.h"
#include "core/words/corpus.h"
#include "core/words/windowsampler.h"
#include "core/words/negativesampler.h"
#include "core/words/model.h"
#include "core/words/trainer.h"
#include "core/words/similarity.h"
#include "core/words/evaluate.h"
#include "core/words/expressions.h"

#include <CLI11/CLI11.hpp>
#include <limits>
#include <unordered_set>

void PrintVocabularyInfo(const Words::Vocabulary& vocabulary) {
    std::cout << "Size: " << vocabulary.getSize() << std::endl;
    std::cout << "First 10 ids:" << std::endl;
    for (Words::TWordId id = 0; id < std::min<Words::TWordId>(10, vocabulary.getSize()); ++id) {
        std::cout << "  - " << vocabulary.getWord(id) << std::endl;
    }
}

void PrintCorpusInfo(const Words::TCorpus& corpus) {
    std::cout << "Corpus size: " << corpus.size() << std::endl;
    std::cout << "First 10 tokens: ";
    for (std::size_t i = 0; i < std::min(10ul, corpus.size()); ++i) {
        std::cout << corpus[i] << " ";
    }
    std::cout << std::endl;
}

void BuildVocabulary(const std::filesystem::path& input, const std::filesystem::path& output, const std::size_t minCount) {
    const auto vocabulary = Words::Vocabulary::Build(input, minCount);
    PrintVocabularyInfo(vocabulary);
    Words::Vocabulary::Save(vocabulary, output);
}

void LoadVocabulary(const std::filesystem::path& input) {
    const auto vocabulary = Words::Vocabulary::Load(input);
    PrintVocabularyInfo(vocabulary);
}

void PrepareCorpus(const std::filesystem::path& input, const std::filesystem::path& output, const Words::Vocabulary& vocabulary) {
    const auto corpus = Words::EncodeCorpus(input, vocabulary);
    PrintCorpusInfo(corpus);
    Words::SaveCorpus(output, corpus);
}

void LoadCorpus(const std::filesystem::path& input) {
    const auto corpus = Words::LoadCorpus(input);
    PrintCorpusInfo(corpus);
}

inline Words::TCorpus SubsampleChunk(
    const Words::TCorpus& corpus,
    const std::size_t from,
    const std::size_t to,
    const Words::Subsampler& sampler,
    XorShift& rng
) {
    Words::TCorpus result;
    result.reserve(to - from);
    for (std::size_t i = from; i < to; ++i) {
        if (sampler.shouldKeep(corpus[i], rng)) {
            result.push_back(corpus[i]);
        }
    }
    return result;
}

void ValidateSubsampler(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& preparedCorpusPath,
    const double sample = 1e-4
) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const auto corpus = Words::LoadCorpus(preparedCorpusPath);
    const Words::Subsampler subsampler(vocabulary, sample);

    std::cout << "sample t = " << sample << '\n';
    std::cout << "Words affected: " << subsampler.getAffectedWordsCount() << " of " << vocabulary.getSize()
              << "  (" << 100.0 * subsampler.getAffectedWordsCount() / vocabulary.getSize() << "%)\n";

    const double expected = subsampler.getExpectedCorpusLength(vocabulary);
    std::cout << "Corpus: " << corpus.size() << " -> ~" << static_cast<std::size_t>(expected)
              << "  (" << std::fixed << std::setprecision(1) << 100.0 * expected / corpus.size()
              << "% survives)\n\n";

    std::cout << "Top-10 keep probabilities:\n";
    for (Words::TWordId id = 0; id < 10 && id < vocabulary.getSize(); ++id) {
        std::cout << "  " << std::setw(8) << vocabulary.getWord(id) << "  share " << std::setw(6)
                  << std::setprecision(3) << 100.0 * vocabulary.getFrequency(id) << "%"
                  << "  keep " << std::setw(6) << 100.0 * subsampler.getKeepProbability(id) << "%\n";
    }

    XorShift rng(12345);
    const auto chunk = SubsampleChunk(corpus, 0, corpus.size(), subsampler, rng);
    std::cout << "\nActual after subsampling: " << chunk.size() << "  ("
              << 100.0 * chunk.size() / corpus.size() << "%)\n";
    std::cout << "(should closely match the estimate above)\n";
    std::cout << "\nBefore:\n  ";
    for (std::size_t i = 0; i < 40 && i < corpus.size(); ++i) {
        std::cout << vocabulary.getWord(corpus[i]) << ' ';
    }
    std::cout << "\n\nAfter:\n  ";
    for (std::size_t i = 0; i < 40 && i < chunk.size(); ++i) {
        std::cout << vocabulary.getWord(chunk[i]) << ' ';
    }
    std::cout << '\n';
}

void VisualSamplerCheck(
    const Words::Vocabulary& vocabulary,
    const Words::TCorpus& corpus,
    const Words::WindowSampler& windowSampler
) {
    std::cout << "=== Toy chunk ===\n";
    Words::TCorpus toy;
    for (std::size_t i = 0; i < 8 && i < corpus.size(); ++i) {
        toy.push_back(corpus[i]);
    }

    std::cout << "chunk: ";
    for (const auto id : toy) {
        std::cout << vocabulary.getWord(id) << ' ';
    }
    std::cout << "\n\n";

    XorShift rng(42);
    auto prevCenter = std::numeric_limits<Words::TWordId>::max();
    windowSampler.forEachPair(toy, rng, [&](const Words::Pair& p) {
        if (p.center != prevCenter) {
            std::cout << "\n  " << vocabulary.getWord(p.center) << " -> ";
            prevCenter = p.center;
        }
        std::cout << vocabulary.getWord(p.context) << ' ';
    });
    std::cout << "\n\n";
}

void CenterContextCheck(const Words::WindowSampler& windowSampler) {
    Words::TCorpus unique;
    for (Words::TWordId i = 0; i < 20; ++i) {
        unique.push_back(i);
    }

    XorShift rng(7);
    bool selfPair = false;
    std::array<int32_t, 20> emitted{};
    windowSampler.forEachPair(unique, rng, [&](const Words::Pair& p) {
        if (p.center == p.context) {
            selfPair = true;
        }
        ++emitted[p.center];
    });

    std::cout << "self-pair check: " << (selfPair ? "FAILED" : "passed") << '\n';
    bool allCovered = true;
    for (const auto count : emitted) {
        if (count == 0) {
            allCovered = false;
        }
    }
    std::cout << "coverage check:  " << (allCovered ? "passed" : "FAILED") << "\n\n";
}

void RealCorpusStatisticsCheck(
    const Words::Vocabulary& vocabulary,
    const Words::TCorpus& corpus,
    const Words::Subsampler& subsampler,
    const Words::WindowSampler& windowSampler
) {
    std::cout << "=== Full pass ===\n";
    XorShift rng(2024);

    std::size_t pairs = 0;
    std::size_t tokens = 0;
    std::vector<std::size_t> asCenter(vocabulary.getSize(), 0);

    auto prevCenter = std::numeric_limits<Words::TWordId>::max();
    GeneratePairs(corpus, subsampler, windowSampler, rng, [&](const Words::Pair& p) {
        ++pairs;
        ++asCenter[p.center];
        if (p.center != prevCenter) {
            ++tokens;
            prevCenter = p.center;
        }
    });

    std::cout << "pairs generated: " << pairs << '\n';
    std::cout << "expected upper bound: "
              << static_cast<std::size_t>(subsampler.getExpectedCorpusLength(vocabulary) * windowSampler.getPairsPerToken())
              << "  (edges make the real count lower)\n\n";

    std::cout << "Most frequent centers after subsampling:\n";
    std::vector<std::pair<std::size_t, Words::TWordId>> byCount;
    byCount.reserve(vocabulary.getSize());
    for (Words::TWordId id = 0; id < vocabulary.getSize(); ++id) {
        byCount.emplace_back(asCenter[id], id);
    }
    std::partial_sort(byCount.begin(), byCount.begin() + 10, byCount.end(), std::greater<>());
    for (std::size_t i = 0; i < 10; ++i) {
        std::cout << "  " << std::setw(10) << vocabulary.getWord(byCount[i].second) << "  " << byCount[i].first << '\n';
    }
}

void PerformSamplerChecks(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& corpusPath
) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const auto corpus = Words::LoadCorpus(corpusPath);
    const Words::Subsampler subsampler(vocabulary, 1e-4);
    const Words::WindowSampler windowSampler(5);

    VisualSamplerCheck(vocabulary, corpus, windowSampler);
    CenterContextCheck(windowSampler);
    RealCorpusStatisticsCheck(vocabulary, corpus, subsampler, windowSampler);
}

void CoverageCheck(const Words::Vocabulary& vocabulary, const Words::NegativeSampler& sampler) {
    std::unordered_set<Words::TWordId> seen;
    XorShift rng(1);
    for (std::size_t i = 0; i < 20'000'000; ++i) {
        seen.insert(sampler.sample(rng));
    }

    std::cout << "reachable words: " << seen.size() << " of " << vocabulary.getSize();
    std::cout << (seen.size() == vocabulary.getSize() ? "  passed" : "  FAILED") << '\n';
}


void DistributionCheck(const Words::Vocabulary& vocabulary, const Words::NegativeSampler& sampler) {
    constexpr std::size_t draws = 20'000'000;
    constexpr double power = 0.75;
    constexpr double minExpectedHits = 50.;
    std::vector<std::size_t> hits(vocabulary.getSize(), 0);
    XorShift rng(2);
    for (std::size_t i = 0; i < draws; ++i) {
        ++hits[sampler.sample(rng)];
    }

    double total = 0.;
    std::vector<double> weights(vocabulary.getSize());
    for (std::size_t id = 0; id < vocabulary.getSize(); ++id) {
        weights[id] = std::pow(static_cast<double>(vocabulary.getCount(id)), power);
        total += weights[id];
    }

    double chiSquare = 0.;
    std::size_t counted = 0;
    double maxAbsZ = 0.;
    std::size_t worstId = 0;

    for (std::size_t id = 0; id < vocabulary.getSize(); ++id) {
        const double p = weights[id] / total;
        const double expected = p * static_cast<double>(draws);
        if (expected < minExpectedHits) {
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

    const double reduced = chiSquare / static_cast<double>(counted);
    const double tolerance = 4. * std::sqrt(2. / static_cast<double>(counted));

    std::cout << "\nwords tested:   " << counted << '\n';
    std::cout << "chi2 / df:      " << std::fixed << std::setprecision(4) << reduced
              << "   (expected 1.0 +- " << tolerance << ")\n";
    std::cout << (std::abs(reduced - 1.) < tolerance ? "  passed" : "  FAILED") << '\n';

    const double expectedMaxZ = std::sqrt(2. * std::log(2. * static_cast<double>(counted)));
    std::cout << "max |z|:        " << maxAbsZ << " (" << vocabulary.getWord(worstId) << ")"
              << "   typical max ~ " << expectedMaxZ << '\n';
}


void FlatteningCheck(const Words::Vocabulary& vocabulary) {
    const auto top = vocabulary.getCount(0);
    const auto rare = vocabulary.getCount(vocabulary.getSize() - 1);

    std::cout << "\nflattening effect (" << vocabulary.getWord(0) << " vs "
              << vocabulary.getWord(vocabulary.getSize() - 1) << "):\n";
    std::cout << "  raw counts ratio:  " << static_cast<double>(top) / rare << '\n';
    std::cout << "  after ^0.75:       "
              << std::pow(static_cast<double>(top), 0.75) / std::pow(static_cast<double>(rare), 0.75)
              << '\n';
}


void ExclusionCheck(const Words::NegativeSampler& sampler) {
    XorShift rng(3);
    constexpr Words::TWordId excluded = 0;
    bool leaked = false;
    for (std::size_t i = 0; i < 1'000'000; ++i) {
        if (sampler.sampleExcluding(excluded, rng) == excluded) {
            leaked = true;
            break;
        }
    }
    std::cout << "\nexclusion check: " << (leaked ? "leaked (raise maxAttempts)" : "passed") << '\n';
}


void PerformNegativeSamplerChecks(const std::filesystem::path& vocabularyPath) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const Words::NegativeSampler sampler(vocabulary);

    std::cout << "=== Negative sampling table ===\n";
    std::cout << "table size: " << sampler.getTableSize() << "  ("
              << sampler.getTableSize() * sizeof(Words::TWordId) / (1024 * 1024) << " MB)\n\n";

    CoverageCheck(vocabulary, sampler);
    DistributionCheck(vocabulary, sampler);
    FlatteningCheck(vocabulary);
    ExclusionCheck(sampler);
}

void PerformModelInitChecks(const std::filesystem::path& vocabularyPath) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);

    Words::ModelConfig config;
    config.dim = 100;

    XorShift rng(1234);
    Words::SGNSModel model(vocabulary, config, rng);

    std::cout << "=== Model init ===\n";
    std::cout << "vocab: " << vocabulary.getSize() << ", dim: " << config.dim << '\n';
    std::cout << "memory: " << model.getBytes() / (1024 * 1024) << " MB\n\n";

    const auto& in = model.getInput();
    const auto& out = model.getOutput();
    const auto dim = config.dim;

    {
        bool allZero = true;
        for (Words::TWordId id = 0; id < vocabulary.getSize(); ++id) {
            const auto* r = out.row(id);
            for (std::size_t i = 0; i < dim; ++i) {
                if (r[i] != 0) {
                    allZero = false;
                }
            }
        }
        std::cout << "output all zero:  " << (allZero ? "passed" : "FAILED") << '\n';
    }

    {
        double sum = 0.;
        double sumSquares = 0.;
        double maxAbs = 0.;
        std::size_t count = 0;

        for (Words::TWordId id = 0; id < vocabulary.getSize(); ++id) {
            const auto* r = in.row(id);
            for (std::size_t i = 0; i < dim; ++i) {
                sum += r[i];
                sumSquares += static_cast<double>(r[i]) * r[i];
                maxAbs = std::max(maxAbs, std::abs(static_cast<double>(r[i])));
                ++count;
            }
        }

        const double mean = sum / count;
        const double stddev = std::sqrt(sumSquares / count - mean * mean);
        const double expectedStddev = (0.5 / dim) / std::sqrt(3.);
        const double bound = 0.5 / dim;

        std::cout << "input mean:       " << std::scientific << std::setprecision(3) << mean
                  << "  (expected ~0)\n";
        std::cout << "input stddev:     " << stddev << "  (expected " << expectedStddev << ")\n";
        std::cout << "input max |x|:    " << maxAbs << "  (bound " << bound << ")\n";
        std::cout << "bound respected:  " << (maxAbs < bound ? "passed" : "FAILED") << '\n';
    }

    {
        bool distinct = true;
        for (Words::TWordId id = 1; id < 100 && id < vocabulary.getSize(); ++id) {
            if (std::equal(in.row(0), in.row(0) + dim, in.row(id))) {
                distinct = false;
            }
        }
        std::cout << "rows distinct:    " << (distinct ? "passed" : "FAILED") << '\n';
    }

    {
        const double score = Words::dot(in.row(0), out.row(1), dim);
        std::cout << "initial score:    " << score << "  (must be exactly 0)\n";
    }

    std::cout << "\nlearning rate schedule:\n";
    for (const double progress : {0., 0.25, 0.5, 0.75, 1.}) {
        const auto processed = static_cast<std::size_t>(progress * 1'000'000);
        std::cout << "  " << std::fixed << std::setprecision(2) << progress << " -> "
                  << std::scientific << model.getLearningRateForStep(processed, 1'000'000) << '\n';
    }
}


namespace {

using TRowSnapshot = std::vector<std::vector<Words::TFloat>>;

TRowSnapshot SaveTouchedRows(
    const Words::SGNSModel& model,
    const Words::Pair& pair,
    const std::vector<Words::TWordId>& negatives
) {
    const auto dim = model.getConfig().dim;
    TRowSnapshot saved;
    saved.reserve(negatives.size() + 2);

    const auto* centerVector = model.getInput().row(pair.center);
    saved.emplace_back(centerVector, centerVector + dim);

    const auto* contextVector = model.getOutput().row(pair.context);
    saved.emplace_back(contextVector, contextVector + dim);

    for (const auto id : negatives) {
        const auto* negativeVector = model.getOutput().row(id);
        saved.emplace_back(negativeVector, negativeVector + dim);
    }
    return saved;
}

void RestoreTouchedRows(
    Words::SGNSModel& model,
    const Words::Pair& pair,
    const std::vector<Words::TWordId>& negatives,
    const TRowSnapshot& saved
) {
    const auto dim = model.getConfig().dim;

    std::copy(saved[0].begin(), saved[0].end(), model.getInputMutable().row(pair.center));
    std::copy(saved[1].begin(), saved[1].end(), model.getOutputMutable().row(pair.context));

    for (std::size_t i = 0; i < negatives.size(); ++i) {
        std::copy(saved[i + 2].begin(), saved[i + 2].end(), model.getOutputMutable().row(negatives[i]));
    }
    (void)dim;
}

double AnalyticGradientOf(
    Words::SGNSModel& model,
    const Words::Pair& pair,
    const std::vector<Words::TWordId>& negatives,
    std::vector<Words::TFloat>& gradient,
    const bool inInputMatrix,
    const Words::TWordId row,
    const std::size_t component
) {
    constexpr double step = 1e-3;

    const auto saved = SaveTouchedRows(model, pair, negatives);

    const auto before = inInputMatrix
        ? model.getInput().row(row)[component]
        : model.getOutput().row(row)[component];

    model.applyUpdate(pair, negatives, step, gradient);

    const auto after = inInputMatrix
        ? model.getInput().row(row)[component]
        : model.getOutput().row(row)[component];

    RestoreTouchedRows(model, pair, negatives, saved);

    return (1. * before - after) / step;
}

double NumericGradientOf(
    Words::SGNSModel& model,
    const Words::Pair& pair,
    const std::vector<Words::TWordId>& negatives,
    const bool inInputMatrix,
    const Words::TWordId row,
    const std::size_t component
) {
    constexpr double eps = 1e-3;

    auto* cell = inInputMatrix
        ? model.getInputMutable().row(row) + component
        : model.getOutputMutable().row(row) + component;

    const auto original = *cell;

    *cell = static_cast<Words::TFloat>(1. * original + eps);
    const double lossPlus = model.computeLoss(pair, negatives);

    *cell = static_cast<Words::TFloat>(1. * original - eps);
    const double lossMinus = model.computeLoss(pair, negatives);

    *cell = original;

    return (lossPlus - lossMinus) / (2. * eps);
}

struct Target {
    bool isInput;
    Words::TWordId row;
    const char* name;
};

}  // namespace


void GradientCheck(const Words::Vocabulary& vocabulary) {
    Words::ModelConfig config;
    config.dim = 16;
    config.negatives = 3;

    XorShift rng(777);
    Words::SGNSModel model(vocabulary, config, rng);
    const Words::NegativeSampler sampler(vocabulary);

    std::vector<Words::TFloat> gradient(config.dim, Words::TFloat{0});
    Words::WorkerContext context(config, 555);

    const Words::Pair pair{7, 42};
    const std::vector<Words::TWordId> negatives = {13, 200, 999};

    for (std::size_t i = 0; i < 30; ++i) {
        model.applyUpdate(pair, negatives, 0.05, gradient);
    }

    for (std::size_t i = 0; i < 200; ++i) {
        const Words::Pair warmup{
            static_cast<Words::TWordId>(rng.nextInteger(1000)),
            static_cast<Words::TWordId>(rng.nextInteger(1000))
        };
        model.trainPair(warmup, 0.05, sampler, context);
    }

    std::cout << "=== Gradient check ===\n";
    std::cout << std::setw(12) << "matrix" << std::setw(8) << "row" << std::setw(6) << "comp"
              << std::setw(14) << "analytic" << std::setw(14) << "numeric"
              << std::setw(12) << "rel.err" << '\n';

    const std::vector<Target> targets = {
        {true,  pair.center,  "W_in[c]"},
        {false, pair.context, "W_out[o]"},
        {false, negatives[0], "W_out[n0]"},
        {false, negatives[2], "W_out[n2]"}
    };

    double worstRelative = 0.;

    for (const auto& target : targets) {
        for (std::size_t component = 0; component < 3; ++component) {
            const double analytic = AnalyticGradientOf(
                model, pair, negatives, gradient, target.isInput, target.row, component);
            const double numeric = NumericGradientOf(
                model, pair, negatives, target.isInput, target.row, component);

            const double scale = std::max({std::abs(analytic), std::abs(numeric), 1e-8});
            const double relative = std::abs(analytic - numeric) / scale;
            worstRelative = std::max(worstRelative, relative);

            std::cout << std::setw(12) << target.name << std::setw(8) << target.row
                      << std::setw(6) << component
                      << std::setw(14) << std::scientific << std::setprecision(4) << analytic
                      << std::setw(14) << numeric
                      << std::setw(12) << relative << '\n';
        }
    }

    std::cout << "\nworst relative error: " << worstRelative
              << (worstRelative < 1e-2 ? "  passed" : "  FAILED") << '\n';
}


void LossBehaviourCheck(const Words::Vocabulary& vocabulary) {
    Words::ModelConfig config;
    config.dim = 50;
    config.negatives = 5;

    XorShift rng(2025);
    Words::SGNSModel model(vocabulary, config, rng);

    const Words::Pair pair{5, 11};
    const std::vector<Words::TWordId> negatives = {100, 200, 300, 400, 500};

    std::cout << "\n=== Loss behaviour ===\n";

    const double initial = model.computeLoss(pair, negatives);
    const double expected = 1. * (config.negatives + 1) * std::log(2.);

    std::cout << "initial loss:  " << std::fixed << std::setprecision(6) << initial
              << "   expected " << expected
              << (std::abs(initial - expected) < 1e-9 ? "  passed" : "  FAILED") << '\n';

    std::cout << "\noverfitting a single pair:\n";

    double previous = initial;
    bool monotone = true;

    std::vector<Words::TFloat> gradient(config.dim, Words::TFloat{0});
    for (std::size_t step = 1; step <= 50; ++step) {
        model.applyUpdate(pair, negatives, 0.05, gradient);
        const double loss = model.computeLoss(pair, negatives);
        if (loss > previous) {
            monotone = false;
        }
        if (step % 10 == 0) {
            std::cout << "  step " << std::setw(3) << step << ":  " << loss << '\n';
        }
        previous = loss;
    }

    std::cout << "monotone decrease: " << (monotone ? "passed" : "FAILED") << '\n';
}


void ModelInitCheck(const Words::Vocabulary& vocabulary) {
    Words::ModelConfig config;
    config.dim = 100;

    XorShift rng(1234);
    const Words::SGNSModel model(vocabulary, config, rng);

    const auto& input = model.getInput();
    const auto& output = model.getOutput();
    const auto dim = config.dim;

    std::cout << "=== Model init ===\n";
    std::cout << "vocab: " << vocabulary.getSize() << ", dim: " << dim << '\n';
    std::cout << "memory: " << model.getBytes() / (1024 * 1024) << " MB\n\n";

    bool allZero = true;
    for (Words::TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const auto* row = output.row(id);
        for (std::size_t i = 0; i < dim; ++i) {
            if (row[i] != 0) {
                allZero = false;
            }
        }
    }
    std::cout << "output all zero:  " << (allZero ? "passed" : "FAILED") << '\n';

    double sum = 0.;
    double sumSquares = 0.;
    double maxAbs = 0.;
    std::size_t count = 0;

    for (Words::TWordId id = 0; id < vocabulary.getSize(); ++id) {
        const auto* row = input.row(id);
        for (std::size_t i = 0; i < dim; ++i) {
            sum += row[i];
            sumSquares += 1. * row[i] * row[i];
            maxAbs = std::max(maxAbs, std::abs(1. * row[i]));
            ++count;
        }
    }

    const double mean = sum / count;
    const double stddev = std::sqrt(sumSquares / count - mean * mean);
    const double expectedStddev = (0.5 / dim) / std::sqrt(3.);
    const double bound = 0.5 / dim;

    std::cout << "input mean:       " << std::scientific << std::setprecision(3) << mean << '\n';
    std::cout << "input stddev:     " << stddev << "  (expected " << expectedStddev << ")\n";
    std::cout << "input max |x|:    " << maxAbs << "  (bound " << bound << ")"
              << (maxAbs < bound ? "  passed" : "  FAILED") << '\n';

    bool distinct = true;
    for (Words::TWordId id = 1; id < 100 && id < vocabulary.getSize(); ++id) {
        if (std::equal(input.row(0), input.row(0) + dim, input.row(id))) {
            distinct = false;
        }
    }
    std::cout << "rows distinct:    " << (distinct ? "passed" : "FAILED") << '\n';
    std::cout << "initial score:    " << Words::dot(input.row(0), output.row(1), dim) << '\n';

    std::cout << "\nlearning rate schedule:\n";
    for (const double progress : {0., 0.25, 0.5, 0.75, 1.}) {
        const auto processed = static_cast<std::size_t>(progress * 1'000'000);
        std::cout << "  " << std::fixed << std::setprecision(2) << progress << " -> "
                  << std::scientific << model.getLearningRateForStep(processed, 1'000'000) << '\n';
    }
    std::cout << '\n';
}


void PerformModelChecks(const std::filesystem::path& vocabularyPath) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);

    ModelInitCheck(vocabulary);
    GradientCheck(vocabulary);
    LossBehaviourCheck(vocabulary);
}


void RunTraining(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& corpusPath,
    const std::filesystem::path& outputPath,
    const Words::ModelConfig& modelConfig,
    const Words::TrainConfig& trainConfig,
    const std::size_t window,
    const double sample
) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const auto corpus = Words::LoadCorpus(corpusPath);

    const Words::Subsampler subsampler(vocabulary, sample);
    const Words::WindowSampler windowSampler(window);
    const Words::NegativeSampler negativeSampler(vocabulary);

    XorShift rng(trainConfig.seed);
    Words::SGNSModel model(vocabulary, modelConfig, rng);

    std::cout << "vocab " << vocabulary.getSize()
              << ", corpus " << corpus.size()
              << ", model " << model.getBytes() / (1024 * 1024) << " MB\n";

    Words::Train(model, corpus, subsampler, windowSampler, negativeSampler, vocabulary, trainConfig);

    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open " + outputPath.string());
    }
    Words::Embeddings::Save(model.getInput(), outputPath);
    std::cout << "saved embeddings to " << outputPath << '\n';
}

void RunNeighbours(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& embeddingsPath,
    const std::string& word,
    const std::size_t count
) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const auto index = Words::EmbeddingIndex::Load(embeddingsPath);

    std::cout << "words " << index.getWords() << ", dim " << index.getDim() << "\n\n";

    if (word.empty()) {
        Words::RunDefaultBattery(vocabulary, index);
    } else {
        Words::PrintNeighbours(vocabulary, index, word, count);
    }
}

void RunEvaluation(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& embeddingsPath,
    const std::filesystem::path& analogiesPath,
    const std::filesystem::path& similarityPath,
    const std::size_t scoreColumn,
    const std::size_t restrictTo,
    const std::size_t threads
) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const auto index = Words::EmbeddingIndex::Load(embeddingsPath);

    std::cout << "words " << index.getWords() << ", dim " << index.getDim() << '\n';

    if (!analogiesPath.empty()) {
        std::cout << "\n=== Analogies ===\n";
        PrintAnalogyReport(
            Words::EvaluateAnalogies(vocabulary, index, analogiesPath, restrictTo, threads));
    }

    if (!similarityPath.empty()) {
        std::cout << "\n=== Similarity ===\n";
        Words::PrintSimilarityReport(
            similarityPath.filename().string(),
            Words::EvaluateSimilarity(vocabulary, index, similarityPath, scoreColumn));
    }
}

void RunVectorOperation(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& embeddingsPath,
    const std::function<void(const Words::Vocabulary&, const Words::EmbeddingIndex&)>& operation
) {
    const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
    const auto index = Words::EmbeddingIndex::Load(embeddingsPath);
    operation(vocabulary, index);
}

int main(int argc, char** argv) {
    CLI::App app{"Words embedder"};
    app.require_subcommand(1);

    std::string inputFilePath;
    std::string outputFilePath;
    std::string vocabularyPath;
    std::string corpusPath;
    double sampleSize = 1e-4;
    std::string embeddingsPath;
    std::size_t dim = 100;
    std::size_t negatives = 5;
    std::size_t window = 5;
    std::size_t epochs = 5;
    double learningRate = 0.025;
    std::string queryWord;
    std::size_t neighbourCount = 10;
    std::size_t threads = 0;
    std::string analogiesPath;
    std::string similarityPath;
    std::size_t scoreColumn = 2;
    std::size_t restrictTo = 30000;
    std::string expression;
    std::string wordList;
    std::string axisExpression;
    std::size_t resultCount = 10;
    std::size_t minCount = 5;

    CLI::App* buildVocabularyCmd = app.add_subcommand("buildvoc", "Build vocabulary");

    buildVocabularyCmd->add_option("--input-file", inputFilePath, "Prepared input file path")
        ->required()->check(CLI::ExistingFile);
    buildVocabularyCmd->add_option("--output-file", outputFilePath, "Output file for built vocabulary")
        ->required();
    buildVocabularyCmd->add_option("--min-count", minCount, "Drop words below this count")
        ->capture_default_str();

    CLI::App* loadVocabularyCmd = app.add_subcommand("loadvoc", "Load vocabulary");
    loadVocabularyCmd->add_option("--input-file", inputFilePath, "Built vocabulary file")
        ->required()->check(CLI::ExistingFile);

    CLI::App* buildCorpusCmd = app.add_subcommand("buildcor", "Build corpus info");
    buildCorpusCmd->add_option("--input-file", inputFilePath, "Prepared corpus file path")
        ->required()->check(CLI::ExistingFile);
    buildCorpusCmd->add_option("--output-file", outputFilePath, "Output file for built corpus")
        ->required();
    buildCorpusCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);

    CLI::App* loadCorpusCmd = app.add_subcommand("loadcor", "Load corpus info");
    loadCorpusCmd->add_option("--input-file", inputFilePath, "Built corpus file path")
        ->required()->check(CLI::ExistingFile);

    CLI::App* validateSubsamplerCmd = app.add_subcommand("validate-subsampler", "Validate subsampler");
    validateSubsamplerCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    validateSubsamplerCmd->add_option("--corpus", corpusPath, "Prepared corpus file")
        ->required()->check(CLI::ExistingFile);
    validateSubsamplerCmd->add_option("--sample", sampleSize, "Sample size")
        ->capture_default_str();

    CLI::App* validateWindowSamplerCmd = app.add_subcommand("validate-window-sampler", "Validate window sampler");
    validateWindowSamplerCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    validateWindowSamplerCmd->add_option("--corpus", corpusPath, "Prepared corpus file")
        ->required()->check(CLI::ExistingFile);

    CLI::App* validateNegativeSamplerCmd = app.add_subcommand("validate-negative-sampler", "Validate negative sampler");
    validateNegativeSamplerCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);

    CLI::App* validateSgnsModelCmd = app.add_subcommand("validate-sgns", "Validate SGNS model");
    validateSgnsModelCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);

    CLI::App* validateGradientsCmd = app.add_subcommand("validate-gradients", "Validate gradients");
    validateGradientsCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);

    CLI::App* trainCmd = app.add_subcommand("train", "Train SGNS embeddings");
    trainCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    trainCmd->add_option("--corpus", corpusPath, "Built corpus path")
        ->required()->check(CLI::ExistingFile);
    trainCmd->add_option("--output-file", embeddingsPath, "Output file for embeddings")
        ->required();
    trainCmd->add_option("--dim", dim, "Embedding dimension")->capture_default_str();
    trainCmd->add_option("--negatives", negatives, "Negatives per pair")->capture_default_str();
    trainCmd->add_option("--window", window, "Max window radius")->capture_default_str();
    trainCmd->add_option("--epochs", epochs, "Epochs")->capture_default_str();
    trainCmd->add_option("--sample", sampleSize, "Subsampling threshold")->capture_default_str();
    trainCmd->add_option("--lr", learningRate, "Initial learning rate")->capture_default_str();
    trainCmd->add_option("--threads", threads, "Worker threads (0 = auto)")->capture_default_str();

    CLI::App* neighboursCmd = app.add_subcommand("neighbours", "Show nearest neighbours");
    neighboursCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    neighboursCmd->add_option("--embeddings", embeddingsPath, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    neighboursCmd->add_option("--word", queryWord, "Query word (empty runs a default battery)");
    neighboursCmd->add_option("--count", neighbourCount, "How many neighbours")->capture_default_str();

    CLI::App* evaluateCmd = app.add_subcommand("evaluate", "Evaluate embeddings");
    evaluateCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    evaluateCmd->add_option("--embeddings", embeddingsPath, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    evaluateCmd->add_option("--analogies", analogiesPath, "questions-words.txt")
        ->check(CLI::ExistingFile);
    evaluateCmd->add_option("--similarity", similarityPath, "WordSim/SimLex file")
        ->check(CLI::ExistingFile);
    evaluateCmd->add_option("--score-column", scoreColumn, "0-based score column")->capture_default_str();
    evaluateCmd->add_option("--restrict-to", restrictTo, "Search top-N words only (0 = all)")->capture_default_str();
    evaluateCmd->add_option("--threads", threads, "Worker threads (0 = auto)")->capture_default_str();

    CLI::App* expressionCmd = app.add_subcommand("expression", "Evaluate a vector expression");
    expressionCmd->add_option("expression", expression, "e.g. \"king - man + woman\"")->required();
    expressionCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    expressionCmd->add_option("--embeddings", embeddingsPath, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    expressionCmd->add_option("--count", resultCount, "How many results")->capture_default_str();

    CLI::App* oddOneCmd = app.add_subcommand("oddone", "Find the odd word out");
    oddOneCmd->add_option("words", wordList, "e.g. \"breakfast cereal lunch dinner\"")->required();
    oddOneCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    oddOneCmd->add_option("--embeddings", embeddingsPath, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);

    CLI::App* axisCmd = app.add_subcommand("axis", "Project words onto a semantic axis");
    axisCmd->add_option("axis", axisExpression, "e.g. \"good - bad\"")->required();
    axisCmd->add_option("--words", wordList, "Words to project (empty scans the vocabulary)");
    axisCmd->add_option("--vocabulary", vocabularyPath, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    axisCmd->add_option("--embeddings", embeddingsPath, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    axisCmd->add_option("--restrict-to", restrictTo, "Scan top-N words (0 = all)")->capture_default_str();
    axisCmd->add_option("--count", resultCount, "How many per end")->capture_default_str();

    CLI11_PARSE(app, argc, argv);
    if (*buildVocabularyCmd) {
        BuildVocabulary(inputFilePath, outputFilePath, minCount);
    } else if (*loadVocabularyCmd) {
        LoadVocabulary(inputFilePath);
    } else if (*buildCorpusCmd) {
        const auto vocabulary = Words::Vocabulary::Load(vocabularyPath);
        PrepareCorpus(inputFilePath, outputFilePath, vocabulary);
    } else if (*loadCorpusCmd) {
        LoadCorpus(inputFilePath);
    } else if (*validateSubsamplerCmd) {
        ValidateSubsampler(vocabularyPath, corpusPath, sampleSize);
    } else if (*validateWindowSamplerCmd) {
        PerformSamplerChecks(vocabularyPath, corpusPath);
    } else if (*validateNegativeSamplerCmd) {
        PerformNegativeSamplerChecks(vocabularyPath);
    } else if (*validateSgnsModelCmd) {
        PerformModelInitChecks(vocabularyPath);
    } else if (*validateGradientsCmd) {
        PerformModelChecks(vocabularyPath);
    } else if (*neighboursCmd) {
        RunNeighbours(vocabularyPath, embeddingsPath, queryWord, neighbourCount);
    } else if (*trainCmd) {
        Words::ModelConfig modelConfig;
        modelConfig.dim = dim;
        modelConfig.negatives = negatives;
        modelConfig.initialLearningRate = learningRate;

        Words::TrainConfig trainConfig;
        trainConfig.epochs = epochs;
        trainConfig.threads = threads;

        RunTraining(vocabularyPath, corpusPath, embeddingsPath, modelConfig, trainConfig, window, sampleSize);
    } else if (*evaluateCmd) {
        RunEvaluation(vocabularyPath, embeddingsPath, analogiesPath, similarityPath, scoreColumn, restrictTo, threads);
    } else if (*expressionCmd) {
        RunVectorOperation(vocabularyPath, embeddingsPath,
            [&](const auto& vocabulary, const auto& index) {
                Words::RunExpression(vocabulary, index, expression, resultCount);
            });
    } else if (*oddOneCmd) {
        RunVectorOperation(vocabularyPath, embeddingsPath,
            [&](const auto& vocabulary, const auto& index) {
                Words::RunOddOne(vocabulary, index, wordList);
            });
    } else if (*axisCmd) {
        RunVectorOperation(vocabularyPath, embeddingsPath,
            [&](const auto& vocabulary, const auto& index) {
                Words::RunAxis(vocabulary, index, axisExpression, wordList, restrictTo, resultCount);
            });
    } else {
        std::cerr << "Error" << std::endl;
    }
    return 0;
}
