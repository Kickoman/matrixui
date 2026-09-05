#pragma once

#include "cli/generator/options.h"

#include <iosfwd>

namespace GeneratorCli {

inline constexpr int kSuccess = 0;
inline constexpr int kLoadFailed = 2;
inline constexpr int kGenerateFailed = 4;

int Generate(std::ostream& out, std::ostream& err, const GenerateOptions& options);
int Train(std::ostream& out, std::ostream& err, const TrainOptions& options);

}  // namespace GeneratorCli
