#include "core/words/utils/inspect.h"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <array>
#include <vector>
#include <algorithm>

namespace Words {

void Inspect(const std::filesystem::path &dump) {
    std::ifstream file(dump);

    std::unordered_map<std::string, std::size_t> words;
    std::size_t symbols = 0;
    std::size_t wordsCount = 0;
    std::string word;
    while (file >> word) {
        words[word] += 1;
        symbols += word.size();
        ++wordsCount;
    }

    std::cout << "Processed " << dump.string() << std::endl;
    std::cout << "Symbols: " << symbols << std::endl;
    std::cout << "Words: " << wordsCount << std::endl;
    std::cout << "Unique words: " << words.size() << std::endl;

    static const std::array<std::size_t, 3> minCounts = {5, 10, 50};
    for (const auto minCount : minCounts) {
        const auto kept = std::count_if(words.cbegin(), words.cend(), [minCount](const auto& el){
            return el.second >= minCount;
        });
        std::cout << "  words with minCount = " << minCount << ": " << kept << std::endl;
    }

    std::cout << "Top-15 by frequency:" << std::endl;
    const auto topN = std::min(15ul, words.size());
    std::vector<std::pair<std::string, std::size_t>> byFreq(words.cbegin(), words.cend());
    std::partial_sort(byFreq.begin(), byFreq.begin() + topN, byFreq.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    for (std::size_t i = 0; i < topN; ++i) {
        const auto& it = byFreq[i];
        const double share = it.second * 1. / wordsCount;
        std::cout << "  " << it.first << " of " << it.second << " (share " << share << ")" << std::endl;
    }
}

}
