#pragma once

#include "core/functions/applier.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace Genetizer {

std::vector<Entry> ParseExpectedCsv(std::istream& in);

void WriteExpectedCsv(std::ostream& out,
                      const std::vector<std::string>& variableNames,
                      const std::vector<Entry>& entries);

}  // namespace Genetizer
