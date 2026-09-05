#pragma once

#include "cli/classifier/options.h"

#include <iosfwd>

namespace ClassifierCli {

// These are a documented contract -- see docs/classifier.md, "Exit codes".
inline constexpr int kSuccess = 0;
inline constexpr int kNoDatasetResolved = 2;
inline constexpr int kInvalidDataset = 3;
inline constexpr int kBadConfig = 4;
inline constexpr int kSizeMismatch = 5;

int Predict(std::ostream& out, std::ostream& err, const PredictOptions& options);
int Train(std::ostream& out, std::ostream& err, const TrainOptions& options);

}  // namespace ClassifierCli
