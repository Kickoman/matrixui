#pragma once

#include "core/words/data/inspect.h"

#include <iosfwd>
#include <string>

namespace Words {

void PrintCorpusStatistics(std::ostream& out, const std::string& name, const CorpusStatistics& statistics);

}  // namespace Words
