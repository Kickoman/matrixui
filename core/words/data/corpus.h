#pragma once

#include "core/words/data/types.h"

#include <iosfwd>
#include <vector>

namespace Words {

class Vocabulary;
using TCorpus = std::vector<TWordId>;

TCorpus EncodeCorpus(std::istream& dump, const Vocabulary& vocabulary);
void SaveCorpus(std::ostream& out, const TCorpus& corpus);
TCorpus LoadCorpus(std::istream& in);

}
