#pragma once

// The body of each subcommand.
//
// Every handler takes the stream to write to and returns whether the command
// succeeded -- the validators return false when a check fails, which is what
// finally gives them a non-zero exit code. They used to print "FAILED" and
// still exit 0, so nothing could gate on them.
//
// These live outside CORE_SOURCES: that list is compiled into the Qt GUI too,
// and there is no reason to pull CLI11 in there.

#include "core/words_cli/options.h"

#include <iosfwd>

namespace WordsCli {

bool Inspect(std::ostream& out, const InspectOptions& options);
bool BuildVocabulary(std::ostream& out, const BuildVocabularyOptions& options);
bool LoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options);
bool BuildCorpus(std::ostream& out, const BuildCorpusOptions& options);
bool LoadCorpus(std::ostream& out, const LoadCorpusOptions& options);

bool ValidateSubsampler(std::ostream& out, const ValidateSubsamplerOptions& options);
bool ValidateWindowSampler(std::ostream& out, const ValidateWindowSamplerOptions& options);
bool ValidateNegativeSampler(std::ostream& out, const ValidateVocabularyOnlyOptions& options);
bool ValidateModel(std::ostream& out, const ValidateVocabularyOnlyOptions& options);
bool ValidateGradients(std::ostream& out, const ValidateVocabularyOnlyOptions& options);

bool Train(std::ostream& out, const TrainOptions& options);
bool Neighbours(std::ostream& out, const NeighboursOptions& options);
bool Evaluate(std::ostream& out, const EvaluateOptions& options);
bool Expression(std::ostream& out, const ExpressionOptions& options);
bool OddOne(std::ostream& out, const OddOneOptions& options);
bool Axis(std::ostream& out, const AxisOptions& options);

}  // namespace WordsCli
