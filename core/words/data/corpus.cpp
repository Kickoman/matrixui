#include "core/words/error.h"
#include "core/words/data/corpus.h"

#include "core/lib/file_stream.h"
#include "core/lib/write.h"
#include "core/words/data/vocabulary.h"

#include <istream>
#include <ostream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace Words {

namespace {

constexpr std::uint32_t CorpusMagic = 0x57435250;
constexpr std::uint32_t CorpusVersion = 1;
constexpr std::size_t CorpusHeaderBytes = 16;
constexpr std::uint64_t LoadedSizeLimit = std::uint64_t{4} << 30;

std::uint64_t ReadCorpusHeader(std::istream& in) {
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
    return size;
}

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

std::uint64_t EncodeCorpusToStream(std::istream& dump, const Vocabulary& vocabulary,
                                   std::ostream& out, const std::size_t bufferTokens) {
    WriteBinaryLE(out, CorpusMagic);
    WriteBinaryLE(out, CorpusVersion);
    WriteBinaryLE(out, std::uint64_t{0});

    TCorpus buffer;
    buffer.reserve(bufferTokens);
    std::uint64_t total = 0;

    std::string word;
    while (dump >> word) {
        const auto id = vocabulary.getId(word);
        if (!id.has_value()) {
            continue;
        }
        buffer.push_back(*id);
        if (buffer.size() >= bufferTokens) {
            WriteBulkLE(out, buffer);
            total += buffer.size();
            buffer.clear();
        }
    }
    WriteBulkLE(out, buffer);
    total += buffer.size();

    out.seekp(8);
    WriteBinaryLE(out, total);
    out.seekp(0, std::ios::end);

    out.flush();
    if (!out) {
        throw IoError("Failed while writing the corpus");
    }
    return total;
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
    const auto size = ReadCorpusHeader(in);
    TCorpus corpus(size);
    ReadBulkLE(in, corpus);
    if (!in) {
        throw IoError("Corpus file is truncated");
    }
    return corpus;
}

std::string_view CorpusStorageName(const CorpusStorage storage) {
    switch (storage) {
        case CorpusStorage::Auto: return "auto";
        case CorpusStorage::Mapped: return "mapped";
        case CorpusStorage::Loaded: return "loaded";
    }
    return "unknown";
}

Corpus::Corpus(TCorpus tokens)
    : owned{std::move(tokens)}
    , base{owned.data()}
    , count{owned.size()}
{ }

Corpus Corpus::Open(const std::filesystem::path& path, const CorpusStorage storage) {
    std::error_code sizeError;
    const std::uint64_t fileSize = std::filesystem::file_size(path, sizeError);
    if (sizeError) {
        throw IoError("Can't read corpus file size: " + path.string());
    }

    const auto declared = Io::ReadFile(
        path, [](std::istream& in) { return ReadCorpusHeader(in); }, std::ios::binary);
    if (fileSize < CorpusHeaderBytes ||
        declared > (fileSize - CorpusHeaderBytes) / sizeof(TWordId)) {
        throw IoError("Corpus file is truncated: " + path.string());
    }

    const std::uint64_t available = Io::AvailableMemoryBytes();
    const bool fits = fileSize < LoadedSizeLimit && available > 0 && fileSize < available / 4;

    CorpusStorage resolved = storage;
    if (resolved == CorpusStorage::Auto) {
        resolved = fits ? CorpusStorage::Loaded : CorpusStorage::Mapped;
    }
    if (resolved == CorpusStorage::Mapped &&
        !(Io::MappedFile::IsSupported() && IsLittleEndian())) {
        resolved = CorpusStorage::Loaded;
    }

    Corpus corpus;
    corpus.storage = resolved;
    if (resolved == CorpusStorage::Loaded) {
        corpus.owned = Io::ReadFile(
            path, [](std::istream& in) { return LoadCorpus(in); }, std::ios::binary);
        corpus.base = corpus.owned.data();
        corpus.count = corpus.owned.size();
    } else {
        corpus.mapping = Io::MappedFile(path);
        corpus.base = reinterpret_cast<const TWordId*>(corpus.mapping.getData() + CorpusHeaderBytes);
        corpus.count = declared;
        if (fits) {
            corpus.mapping.adviseWillNeed();
        } else {
            corpus.mapping.adviseSequential();
        }
    }
    return corpus;
}

Corpus::Corpus(Corpus&& other) noexcept
    : owned{std::move(other.owned)}
    , mapping{std::move(other.mapping)}
    , base{std::exchange(other.base, nullptr)}
    , count{std::exchange(other.count, 0)}
    , storage{other.storage}
{
    if (storage == CorpusStorage::Loaded) {
        base = owned.data();
    }
}

Corpus& Corpus::operator=(Corpus&& other) noexcept {
    if (this != &other) {
        owned = std::move(other.owned);
        mapping = std::move(other.mapping);
        base = std::exchange(other.base, nullptr);
        count = std::exchange(other.count, 0);
        storage = other.storage;
        if (storage == CorpusStorage::Loaded) {
            base = owned.data();
        }
    }
    return *this;
}

}
