#pragma once

#include "core/lib/mapped_file.h"
#include "core/words/data/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string_view>
#include <vector>

namespace Words {

class Vocabulary;
using TCorpus = std::vector<TWordId>;

TCorpus EncodeCorpus(std::istream& dump, const Vocabulary& vocabulary);
void SaveCorpus(std::ostream& out, const TCorpus& corpus);
TCorpus LoadCorpus(std::istream& in);
std::uint64_t EncodeCorpusToStream(std::istream& dump, const Vocabulary& vocabulary,
                                   std::ostream& out, std::size_t bufferTokens = 1 << 18);

enum class CorpusStorage { Auto, Mapped, Loaded };

std::string_view CorpusStorageName(CorpusStorage storage);

class Corpus {
public:
    Corpus() = default;
    explicit Corpus(TCorpus tokens);

    static Corpus Open(const std::filesystem::path& path,
                       CorpusStorage storage = CorpusStorage::Auto);

    std::size_t size() const { return count; }
    const TWordId* data() const { return base; }
    TWordId operator[](const std::size_t index) const { return base[index]; }
    const TWordId* begin() const { return base; }
    const TWordId* end() const { return base + count; }
    CorpusStorage getStorage() const { return storage; }

    Corpus(Corpus&& other) noexcept;
    Corpus& operator=(Corpus&& other) noexcept;
    Corpus(const Corpus&) = delete;
    Corpus& operator=(const Corpus&) = delete;

private:
    TCorpus owned;
    Io::MappedFile mapping;
    const TWordId* base{nullptr};
    std::size_t count{0};
    CorpusStorage storage{CorpusStorage::Loaded};
};

}
