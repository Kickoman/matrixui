#pragma once

#include "cli/functions/options.h"

#include <iosfwd>

namespace FunctionsCli {

inline constexpr int kSuccess = 0;
inline constexpr int kFailure = 1;
inline constexpr int kInternalError = 2;

int Run(std::ostream& out, std::ostream& err, const RunOptions& options);

}  // namespace FunctionsCli
