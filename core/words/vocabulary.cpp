#include "core/words/error.h"
#include "core/words/vocabulary.h"
#include "core/lib/write.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace Words {

Vocabulary Vocabulary::Build(const std::filesystem::path &dump, const std::size_t minCount) {
    std::ifstream file(dump);
    if (!file) {
        throw IoError("Cannot open file: " + dump.string());
    }

    std::unordered_map<std::string, std::size_t> frequencies;
    frequencies.reserve(1 << 20);  // ??? why

    std::size_t rawTokens = 0;
    std::string word;
    while (file >> word) {
        ++frequencies[word];
        ++rawTokens;
    }

    struct WordInfo {
        std::string word;
        std::size_t occurrences;
    };
    std::vector<WordInfo> kept;
    kept.reserve(frequencies.size());
    for (auto& [word, count] : frequencies) {
        if (count >= minCount) {
            kept.emplace_back(std::move(word), count);
        }
    }

    std::sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) {
        return a.word != b.word ? a.occurrences > b.occurrences : a.word < b.word;
    });

    if (kept.size() > std::numeric_limits<TWordId>::max()) {
        throw VocabularyError("vocabulary too large for TWordId");
    }

    Vocabulary vocabulary;
    vocabulary.id2word.reserve(kept.size());
    vocabulary.counts.reserve(kept.size());
    vocabulary.word2id.reserve(kept.size());

    for (TWordId i = 0; i < kept.size(); ++i) {
        auto& info = kept[i];
        vocabulary.word2id.emplace(info.word, i);
        vocabulary.id2word.push_back(std::move(info.word));
        vocabulary.counts.push_back(info.occurrences);
        vocabulary.keptTokens += info.occurrences;
    }
    vocabulary.rawTokens = rawTokens;
    return vocabulary;
}

std::optional<TWordId> Vocabulary::getId(const std::string& word) const {
    const auto it = word2id.find(word);
    if (it == word2id.end()) {
        return std::nullopt;
    }
    return it->second;
}

double Vocabulary::getFrequency(const TWordId id) const {
    return 1. * counts[id] / keptTokens;
}

Vocabulary Vocabulary::Load(const std::filesystem::path &path) {
    // Binary, to match Save. This used to open in text mode, which happens to
    // be identical on POSIX but mangles the payload on Windows.
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError("Can't open file for reading: " + path.string());
    }

    std::uint64_t size = 0;
    std::size_t raw = 0;
    std::size_t kept = 0;

    ReadBinaryLE(file, size);
    ReadBinaryLE(file, raw);
    ReadBinaryLE(file, kept);

    // Unchecked before: anything that opened -- /dev/null, a text file, a
    // truncated artifact -- came back as a silent zero-word vocabulary that
    // every downstream command then treated as valid.
    if (!file) {
        throw IoError("Not a vocabulary file (header is truncated): " + path.string());
    }
    if (size == 0) {
        throw VocabularyError("Vocabulary file contains no words: " + path.string());
    }
    if (size > std::numeric_limits<TWordId>::max()) {
        throw VocabularyError("vocabulary too large for TWordId: " + path.string());
    }

    Vocabulary vocabulary;
    vocabulary.rawTokens = raw;
    vocabulary.keptTokens = kept;
    vocabulary.id2word.reserve(size);
    vocabulary.counts.reserve(size);
    vocabulary.word2id.reserve(size);

    for (TWordId i = 0; i < size; ++i) {
        std::size_t length = 0;
        ReadBinaryLE(file, length);
        std::string word(length, '\0');
        file.read(word.data(), length);
        std::size_t count = 0;
        ReadBinaryLE(file, count);

        if (!file) {
            throw IoError("Vocabulary file is truncated: " + path.string());
        }

        vocabulary.word2id.emplace(word, i);
        vocabulary.id2word.emplace_back(std::move(word));
        vocabulary.counts.push_back(count);
    }
    return vocabulary;
}

void Vocabulary::Save(const Vocabulary &vocabulary, const std::filesystem::path &path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw IoError("Can't open file for writing: " + path.string());
    }
    WriteBinaryLE(file, static_cast<std::uint64_t>(vocabulary.getSize()));
    WriteBinaryLE(file, vocabulary.rawTokens);
    WriteBinaryLE(file, vocabulary.keptTokens);
    for (TWordId i = 0; i < vocabulary.getSize(); ++i) {
        WriteBinaryLE(file, vocabulary.id2word[i].size());
        file.write(vocabulary.id2word[i].data(), vocabulary.id2word[i].size());
        WriteBinaryLE(file, vocabulary.counts[i]);
    }

    file.flush();
    if (!file) {
        throw IoError("Failed while writing the vocabulary to " + path.string());
    }
}

}
