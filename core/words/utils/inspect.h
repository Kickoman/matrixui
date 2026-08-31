#pragma once

// A quick look at a raw text dump, before any vocabulary is built.
//
// Useful for choosing --min-count: it reports how many words would survive at
// a few thresholds.

#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace Words {

struct CorpusStatistics {
    std::filesystem::path path;
    std::size_t symbols{0};
    std::size_t totalWords{0};
    std::size_t uniqueWords{0};

    // minCount -> how many distinct words would survive it.
    std::vector<std::pair<std::size_t, std::size_t>> survivorsByMinCount;

    struct Entry {
        std::string word;
        std::size_t count{0};
        double share{0.};
    };
    std::vector<Entry> topByFrequency;
};

CorpusStatistics InspectDump(const std::filesystem::path& dump, std::size_t topN = 15);

}  // namespace Words
