#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace Words {

struct CorpusStatistics {
    std::size_t symbols{0};
    std::size_t totalWords{0};
    std::size_t uniqueWords{0};

    std::vector<std::pair<std::size_t, std::size_t>> survivorsByMinCount;

    struct Entry {
        std::string word;
        std::size_t count{0};
        double share{0.};
    };
    std::vector<Entry> topByFrequency;
};

CorpusStatistics InspectDump(std::istream& dump, std::size_t topN = 15);

}  // namespace Words
