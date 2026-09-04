#include "core/words/error.h"
#include "core/words/data/corpus.h"

#include "core/lib/write.h"
#include "core/words/data/vocabulary.h"

#include <istream>
#include <ostream>
#include <stdexcept>

namespace Words {

namespace {

constexpr std::uint32_t CorpusMagic = 0x57435250;
constexpr std::uint32_t CorpusVersion = 1;

}

TCorpus EncodeCorpus(std::istream &dump, const Vocabulary &vocabulary) {
    TCorpus corpus;
    corpus.reserve(vocabulary.getKeptTokens());

    std::string word;
    while (dump >> word) {
        const auto id = vocabulary.getId(word);
        if (id.has_value()) {
            corpus.push_back(*id);
        }
    }
    return corpus;
}

void SaveCorpus(std::ostream &out, const TCorpus &corpus) {
    WriteBinaryLE(out, CorpusMagic);
    WriteBinaryLE(out, CorpusVersion);
    WriteBinaryLE(out, static_cast<std::uint64_t>(corpus.size()));
    WriteBulkLE(out, corpus);

    out.flush();
    if (!out) {
        throw IoError("Failed while writing the corpus");
    }
}

TCorpus LoadCorpus(std::istream& in) {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    ReadBinaryLE(in, magic);
    ReadBinaryLE(in, version);
    if (magic != CorpusMagic) {
        throw IoError("Not a corpus file or built by an older version (rebuild it with buildcor)");
    }
    if (version != CorpusVersion) {
        throw IoError("Unsupported corpus version");
    }

    std::uint64_t size = 0;
    ReadBinaryLE(in, size);
    TCorpus corpus(size);
    ReadBulkLE(in, corpus);
    return corpus;
}

}
