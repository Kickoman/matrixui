#include "core/words/utils/inspect.h"

#include "core/words/error.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <unordered_map>

namespace Words {

CorpusStatistics InspectDump(const std::filesystem::path& dump, const std::size_t topN) {
    std::ifstream file(dump);
    if (!file) {
        throw IoError("Can't open file for reading: " + dump.string());
    }

    std::unordered_map<std::string, std::size_t> words;
    CorpusStatistics statistics;
    statistics.path = dump;

    std::string word;
    while (file >> word) {
        words[word] += 1;
        statistics.symbols += word.size();
        ++statistics.totalWords;
    }
    statistics.uniqueWords = words.size();

    static constexpr std::array<std::size_t, 3> minCounts = {5, 10, 50};
    for (const auto minCount : minCounts) {
        const auto kept = std::count_if(words.cbegin(), words.cend(),
            [minCount](const auto& entry) { return entry.second >= minCount; });
        statistics.survivorsByMinCount.emplace_back(minCount, static_cast<std::size_t>(kept));
    }

    const auto shown = std::min(topN, words.size());
    std::vector<std::pair<std::string, std::size_t>> byFrequency(words.cbegin(), words.cend());
    std::partial_sort(byFrequency.begin(), byFrequency.begin() + shown, byFrequency.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    statistics.topByFrequency.reserve(shown);
    for (std::size_t i = 0; i < shown; ++i) {
        const auto& [text, count] = byFrequency[i];
        statistics.topByFrequency.push_back({
            text,
            count,
            statistics.totalWords > 0
                ? static_cast<double>(count) / static_cast<double>(statistics.totalWords)
                : 0.,
        });
    }

    return statistics;
}

}  // namespace Words
