#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <optional>

namespace Words {

class Vocabulary {
public:

    static Vocabulary Build(const std::filesystem::path& dump, std::size_t minCount = 5);

    std::optional<std::size_t> getId(const std::string& word) const;
    const std::string& getWord(std::size_t id) const { return id2word[id]; }

    std::size_t getCount(std::size_t id) const { return counts[id]; }
    std::size_t getSize() const { return id2word.size(); }
    std::size_t getKeptTokens() const { return keptTokens; }
    std::size_t getRawTokens() const { return rawTokens; }
    double getFrequency(std::size_t id) const;

    static Vocabulary Load(const std::filesystem::path& vocabulary);
    static void Save(const Vocabulary& vocabulary, const std::filesystem::path& path);
private:
    std::vector<std::string> id2word;
    std::unordered_map<std::string, std::size_t> word2id;
    std::vector<std::size_t> counts;
    std::size_t rawTokens{0};
    std::size_t keptTokens{0};
};

}
