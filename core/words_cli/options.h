#pragma once

// One option struct per subcommand.
//
// main_words.cpp used to declare ~25 mutable locals shared by all fifteen
// subcommands, so `--input-file` meant a different thing depending on which
// one ran. Each command now binds to its own fields.

#include "core/words/config.h"

#include <cstddef>
#include <filesystem>
#include <string>

namespace WordsCli {

struct BuildVocabularyOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    std::size_t minCount{5};
};

struct InspectOptions {
    std::filesystem::path input;
    std::size_t topN{15};
};

struct LoadVocabularyOptions {
    std::filesystem::path input;
};

struct BuildCorpusOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path vocabulary;
};

struct LoadCorpusOptions {
    std::filesystem::path input;
};

struct TrainOptions {
    std::filesystem::path vocabulary;
    std::filesystem::path corpus;
    std::filesystem::path output;
    Words::WordsConfig config;
};

struct NeighboursOptions {
    std::filesystem::path vocabulary;
    std::filesystem::path embeddings;
    std::string word;
    std::size_t count{10};
};

struct EvaluateOptions {
    std::filesystem::path vocabulary;
    std::filesystem::path embeddings;
    std::filesystem::path analogies;
    std::filesystem::path similarity;
    std::size_t scoreColumn{2};
    std::size_t restrictTo{30000};
    std::size_t threads{0};
};

struct ExpressionOptions {
    std::filesystem::path vocabulary;
    std::filesystem::path embeddings;
    std::string expression;
    std::size_t count{10};
};

struct OddOneOptions {
    std::filesystem::path vocabulary;
    std::filesystem::path embeddings;
    std::string words;
};

struct AxisOptions {
    std::filesystem::path vocabulary;
    std::filesystem::path embeddings;
    std::string axis;
    std::string words;
    std::size_t restrictTo{30000};
    std::size_t count{10};
};

}  // namespace WordsCli
