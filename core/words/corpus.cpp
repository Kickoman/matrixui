#include "corpus.h"

#include "core/lib/write.h"
#include "core/words/vocabulary.h"

#include <fstream>
#include <stdexcept>

namespace Words {

TCorpus EncodeCorpus(const std::filesystem::path &dump, const Vocabulary &vocabulary) {
    std::ifstream file(dump);
    if (!file) {
        throw std::runtime_error("Can't open file for reading: " + dump.string());
    }

    TCorpus corpus;
    corpus.reserve(vocabulary.getKeptTokens());

    std::string word;
    while (file >> word) {
        const auto id = vocabulary.getId(word);
        if (id.has_value()) {
            corpus.push_back(*id);
        }
    }
    return corpus;
}

void SaveCorpus(const std::filesystem::path &path, const TCorpus &corpus) {
    std::ofstream file(path, std::ios::binary);
    WriteBinaryLE(file, corpus.size());
    WriteBulkLE(file, corpus);
}

TCorpus LoadCorpus(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Can't open file for reading: " + path.string());
    }
    std::size_t size = 0;
    ReadBinaryLE(file, size);
    TCorpus corpus(size);
    ReadBulkLE(file, corpus);
    return corpus;
}

}
