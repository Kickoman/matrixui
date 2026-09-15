#include "core/words/error.h"
#include "core/words/data/vocabulary.h"
#include "core/lib/write.h"

#include <istream>
#include <ostream>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace Words {

Vocabulary Vocabulary::Build(std::istream &dump, const std::size_t minCount,
                             const std::size_t pruneThreshold, VocabularyBuildStats* stats) {
    std::unordered_map<std::string, std::size_t> frequencies;
    frequencies.reserve(1 << 20);

    std::size_t rawTokens = 0;
    std::size_t minReduce = 1;
    std::size_t pruneRuns = 0;
    std::string word;
    while (dump >> word) {
        ++frequencies[word];
        ++rawTokens;
        if (pruneThreshold > 0 && frequencies.size() > pruneThreshold) {
            std::erase_if(frequencies, [minReduce](const auto& entry) {
                return entry.second <= minReduce;
            });
            ++minReduce;
            ++pruneRuns;
        }
    }

    if (stats != nullptr) {
        stats->pruneRuns = pruneRuns;
        stats->finalMinReduce = pruneRuns > 0 ? minReduce - 1 : 0;
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
        return a.occurrences != b.occurrences ? a.occurrences > b.occurrences : a.word < b.word;
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

Vocabulary Vocabulary::Load(std::istream &in) {
    std::uint64_t size = 0;
    std::size_t raw = 0;
    std::size_t kept = 0;

    ReadBinaryLE(in, size);
    ReadBinaryLE(in, raw);
    ReadBinaryLE(in, kept);

    if (!in) {
        throw IoError("Not a vocabulary file (header is truncated)");
    }
    if (size == 0) {
        throw VocabularyError("Vocabulary file contains no words");
    }
    if (size > std::numeric_limits<TWordId>::max()) {
        throw VocabularyError("vocabulary too large for TWordId");
    }

    Vocabulary vocabulary;
    vocabulary.rawTokens = raw;
    vocabulary.keptTokens = kept;
    vocabulary.id2word.reserve(size);
    vocabulary.counts.reserve(size);
    vocabulary.word2id.reserve(size);

    for (TWordId i = 0; i < size; ++i) {
        std::size_t length = 0;
        ReadBinaryLE(in, length);
        std::string word(length, '\0');
        in.read(word.data(), length);
        std::size_t count = 0;
        ReadBinaryLE(in, count);

        if (!in) {
            throw IoError("Vocabulary file is truncated");
        }

        vocabulary.word2id.emplace(word, i);
        vocabulary.id2word.emplace_back(std::move(word));
        vocabulary.counts.push_back(count);
    }
    return vocabulary;
}

void Vocabulary::Save(std::ostream &out, const Vocabulary &vocabulary) {
    WriteBinaryLE(out, static_cast<std::uint64_t>(vocabulary.getSize()));
    WriteBinaryLE(out, vocabulary.rawTokens);
    WriteBinaryLE(out, vocabulary.keptTokens);
    for (TWordId i = 0; i < vocabulary.getSize(); ++i) {
        WriteBinaryLE(out, vocabulary.id2word[i].size());
        out.write(vocabulary.id2word[i].data(), vocabulary.id2word[i].size());
        WriteBinaryLE(out, vocabulary.counts[i]);
    }

    out.flush();
    if (!out) {
        throw IoError("Failed while writing the vocabulary");
    }
}

}
