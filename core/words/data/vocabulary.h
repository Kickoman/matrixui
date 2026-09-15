#pragma once

#include "core/words/data/types.h"

#include <iosfwd>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

namespace Words {

struct VocabularyBuildStats {
    std::size_t pruneRuns{0};
    std::size_t finalMinReduce{0};
};

class Vocabulary {
public:

    static constexpr std::size_t DefaultPruneThreshold = 20'000'000;

    static Vocabulary Build(std::istream& dump, std::size_t minCount = 5,
                            std::size_t pruneThreshold = DefaultPruneThreshold,
                            VocabularyBuildStats* stats = nullptr);

    std::optional<TWordId> getId(const std::string& word) const;
    const std::string& getWord(TWordId id) const { return id2word[id]; }

    std::size_t getCount(TWordId id) const { return counts[id]; }
    TWordId getSize() const { return static_cast<TWordId>(id2word.size()); }
    std::size_t getKeptTokens() const { return keptTokens; }
    std::size_t getRawTokens() const { return rawTokens; }
    double getFrequency(TWordId id) const;

    static Vocabulary Load(std::istream& in);
    static void Save(std::ostream& out, const Vocabulary& vocabulary);
private:
    std::vector<std::string> id2word;
    std::unordered_map<std::string, TWordId> word2id;
    std::vector<std::size_t> counts;
    std::size_t rawTokens{0};
    std::size_t keptTokens{0};
};

}
