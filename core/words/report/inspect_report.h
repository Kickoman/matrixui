#pragma once

#include "core/words/data/inspect.h"

#include <iosfwd>

namespace Words {

void PrintCorpusStatistics(std::ostream& out, const CorpusStatistics& statistics);

}  // namespace Words
