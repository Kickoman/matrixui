#pragma once

// The body of each subcommand.
//
// Every handler takes the stream to write to. Failures are reported by
// throwing Words::Error (missing/broken files, impossible configs), which
// main() turns into a message and a non-zero exit code; bad user input such as
// an unknown word comes back as data inside the printed report and is not a
// failure.
//
// These live outside CORE_SOURCES: that list is compiled into the Qt GUI too,
// and there is no reason to pull CLI11 in there.

#include "core/words_cli/options.h"

#include <iosfwd>

namespace WordsCli {

void Inspect(std::ostream& out, const InspectOptions& options);
void BuildVocabulary(std::ostream& out, const BuildVocabularyOptions& options);
void LoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options);
void BuildCorpus(std::ostream& out, const BuildCorpusOptions& options);
void LoadCorpus(std::ostream& out, const LoadCorpusOptions& options);

void Train(std::ostream& out, const TrainOptions& options);
void Neighbours(std::ostream& out, const NeighboursOptions& options);
void Evaluate(std::ostream& out, const EvaluateOptions& options);
void Expression(std::ostream& out, const ExpressionOptions& options);
void OddOne(std::ostream& out, const OddOneOptions& options);
void Axis(std::ostream& out, const AxisOptions& options);

}  // namespace WordsCli
