#pragma once

#include "cli/serving/options.h"

#include <iosfwd>

namespace ServingCli {

inline constexpr int kSuccess = 0;
inline constexpr int kIncomplete = 1;
inline constexpr int kUnusableRoot = 2;
inline constexpr int kBadOption = 3;

int List(std::ostream& out, std::ostream& err, const ListOptions& options);

}  // namespace ServingCli
