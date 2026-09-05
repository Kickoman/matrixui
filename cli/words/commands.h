#pragma once

#include "cli/words/options.h"

#include <iosfwd>

namespace WordsCli {

// Exit codes: 0 success, 1 a reported error, 2 a bug. Parse failures never
// reach these functions -- CLI11 owns them: 106 for a missing required
// option or subcommand, 105 for a failed file check, 109 for an unknown flag.
inline constexpr int kSuccess = 0;
inline constexpr int kFailure = 1;
inline constexpr int kInternalError = 2;

int Inspect(std::ostream& out, std::ostream& err, const InspectOptions& options);
int BuildVocabulary(std::ostream& out, std::ostream& err, const BuildVocabularyOptions& options);
int LoadVocabulary(std::ostream& out, std::ostream& err, const LoadVocabularyOptions& options);
int BuildCorpus(std::ostream& out, std::ostream& err, const BuildCorpusOptions& options);
int LoadCorpus(std::ostream& out, std::ostream& err, const LoadCorpusOptions& options);

int Train(std::ostream& out, std::ostream& err, const TrainOptions& options);
int Neighbours(std::ostream& out, std::ostream& err, const NeighboursOptions& options);
int Evaluate(std::ostream& out, std::ostream& err, const EvaluateOptions& options);
int Expression(std::ostream& out, std::ostream& err, const ExpressionOptions& options);
int OddOne(std::ostream& out, std::ostream& err, const OddOneOptions& options);
int Axis(std::ostream& out, std::ostream& err, const AxisOptions& options);

}  // namespace WordsCli
