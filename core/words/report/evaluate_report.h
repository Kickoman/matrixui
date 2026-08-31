#pragma once

#include "core/words/evaluate.h"

#include <iosfwd>
#include <string>

namespace Words {

void PrintAnalogyReport(std::ostream& out, const AnalogyReport& report);
void PrintSimilarityReport(std::ostream& out, const std::string& name, const SimilarityReport& report);

}  // namespace Words
