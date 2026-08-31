#include <iostream>

#include "core/lib/random.h"
#include "core/words/subsampler.h"
#include "core/words/vocabulary.h"
#include "core/words/corpus.h"
#include "core/words/windowsampler.h"
#include "core/words/negativesampler.h"

#include <CLI11/CLI11.hpp>
#include <unordered_set>

void PrintVocabularyInfo(const Words::Vocabulary& vocabulary) {
    std::cout << "Size: " << vocabulary.getSize() << std::endl;
    std::cout << "First 10 ids:" << std::endl;
    for (std::size_t id = 0; id < std::min(10ul, vocabulary.getSize()); ++id) {
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

void BuildVocabulary(const std::filesystem::path& input, const std::filesystem::path& output) {
    const auto vocabulary = Words::Vocabulary::Build(input);
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
    for (int32_t id = 0; id < 10 && id < static_cast<int32_t>(vocabulary.getSize()); ++id) {
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
    int32_t prevCenter = -1;
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
    for (int32_t i = 0; i < 20; ++i) {
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

    int32_t prevCenter = -1;
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
    std::vector<std::pair<std::size_t, int32_t>> byCount;
    byCount.reserve(vocabulary.getSize());
    for (std::size_t id = 0; id < vocabulary.getSize(); ++id) {
        byCount.emplace_back(asCenter[id], static_cast<int32_t>(id));
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
    std::unordered_set<std::size_t> seen;
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

    std::vector<std::size_t> hits(vocabulary.getSize(), 0);
    XorShift rng(2);
    for (std::size_t i = 0; i < draws; ++i) {
        ++hits[sampler.sample(rng)];
    }

    double total = 0.;
    for (std::size_t id = 0; id < vocabulary.getSize(); ++id) {
        total += std::pow(static_cast<double>(vocabulary.getCount(id)), power);
    }

    std::cout << "\n" << std::setw(10) << "word" << std::setw(12) << "expected"
              << std::setw(12) << "actual" << std::setw(10) << "ratio" << '\n';

    double worstRatio = 1.;
    for (std::size_t id = 0; id < vocabulary.getSize(); ++id) {
        const double expected = std::pow(static_cast<double>(vocabulary.getCount(id)), power) / total;
        const double actual = static_cast<double>(hits[id]) / draws;

        if (expected * draws < 1000) {
            continue;
        }
        const double ratio = actual / expected;
        worstRatio = std::max(worstRatio, ratio > 1. ? ratio : 1. / ratio);

        if (id < 5) {
            std::cout << std::setw(10) << vocabulary.getWord(id)
                      << std::setw(12) << std::fixed << std::setprecision(6) << expected
                      << std::setw(12) << actual
                      << std::setw(10) << std::setprecision(3) << ratio << '\n';
        }
    }

    std::cout << "\nworst ratio among well-sampled words: " << worstRatio;
    std::cout << (worstRatio < 1.05 ? "  passed" : "  FAILED") << '\n';
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
    constexpr std::size_t excluded = 0;
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
              << sampler.getTableSize() * sizeof(std::size_t) / (1024 * 1024) << " MB)\n\n";

    CoverageCheck(vocabulary, sampler);
    DistributionCheck(vocabulary, sampler);
    FlatteningCheck(vocabulary);
    ExclusionCheck(sampler);
}


int main(int argc, char** argv) {
    CLI::App app{"Words embedder"};
    app.require_subcommand(1);

    std::string inputFilePath;
    std::string outputFilePath;
    std::string vocabularyPath;
    std::string corpusPath;

    double sampleSize = 1e-4;

    CLI::App* buildVocabularyCmd = app.add_subcommand("buildvoc", "Build vocabulary");

    buildVocabularyCmd->add_option("--input-file", inputFilePath, "Prepared input file path")
        ->required()->check(CLI::ExistingFile);
    buildVocabularyCmd->add_option("--output-file", outputFilePath, "Output file for built vocabulary")
        ->required();

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

    CLI11_PARSE(app, argc, argv);
    if (*buildVocabularyCmd) {
        BuildVocabulary(inputFilePath, outputFilePath);
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
    } else {
        std::cerr << "Error" << std::endl;
    }
    return 0;
}
