#include "corpus.h"

#include "core/lib/write.h"
#include "core/words/vocabulary.h"

#include <fstream>
#include <stdexcept>

namespace Words {

namespace {

constexpr std::uint32_t CorpusMagic = 0x57435250;  // 'WCRP'
constexpr std::uint32_t CorpusVersion = 1;

}

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
    WriteBinaryLE(file, CorpusMagic);
    WriteBinaryLE(file, CorpusVersion);
    WriteBinaryLE(file, static_cast<std::uint64_t>(corpus.size()));
    WriteBulkLE(file, corpus);
}

TCorpus LoadCorpus(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Can't open file for reading: " + path.string());
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    ReadBinaryLE(file, magic);
    ReadBinaryLE(file, version);
    if (magic != CorpusMagic) {
        throw std::runtime_error(
            "Not a corpus file or built by an older version: " + path.string()
            + ". Rebuild it with buildcor."
        );
    }
    if (version != CorpusVersion) {
        throw std::runtime_error("Unsupported corpus version: " + path.string());
    }

    std::uint64_t size = 0;
    ReadBinaryLE(file, size);
    TCorpus corpus(size);
    ReadBulkLE(file, corpus);
    return corpus;
}

}
