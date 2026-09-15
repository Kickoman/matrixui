#pragma once

#include "core/functions/applier.h"

#include <iosfwd>
#include <vector>

namespace FunctionsCli {

std::vector<Genetizer::Entry> ParseExpectedCsv(std::istream& in);

}  // namespace FunctionsCli
