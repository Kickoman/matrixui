#pragma once

#include <vector>
#include <filesystem>

namespace Words {

class Vocabulary;
using TCorpus = std::vector<std::size_t>;

TCorpus EncodeCorpus(const std::filesystem::path& dump, const Vocabulary& vocabulary);
void SaveCorpus(const std::filesystem::path& path, const TCorpus& corpus);
TCorpus LoadCorpus(const std::filesystem::path& path);

}
