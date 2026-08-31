#pragma once

#include "core/words/types.h"

#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <optional>

namespace Words {

class Vocabulary {
public:

    static Vocabulary Build(const std::filesystem::path& dump, std::size_t minCount = 5);

    std::optional<TWordId> getId(const std::string& word) const;
    const std::string& getWord(TWordId id) const { return id2word[id]; }

    std::size_t getCount(TWordId id) const { return counts[id]; }
    TWordId getSize() const { return static_cast<TWordId>(id2word.size()); }
    std::size_t getKeptTokens() const { return keptTokens; }
    std::size_t getRawTokens() const { return rawTokens; }
    double getFrequency(TWordId id) const;

    static Vocabulary Load(const std::filesystem::path& vocabulary);
    static void Save(const Vocabulary& vocabulary, const std::filesystem::path& path);
private:
    std::vector<std::string> id2word;
    std::unordered_map<std::string, TWordId> word2id;
    std::vector<std::size_t> counts;
    std::size_t rawTokens{0};
    std::size_t keptTokens{0};
};

}
