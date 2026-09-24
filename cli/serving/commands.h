#pragma once

#include "cli/serving/options.h"

#include <iosfwd>

namespace ServingCli {

inline constexpr int kSuccess = 0;
inline constexpr int kIncomplete = 1;
inline constexpr int kUnusableRoot = 2;
inline constexpr int kBadOption = 3;
inline constexpr int kCannotBind = 4;

int List(std::ostream& out, std::ostream& err, const ListOptions& options);
int Serve(std::ostream& out, std::ostream& err, const ServeOptions& options);

}  // namespace ServingCli
