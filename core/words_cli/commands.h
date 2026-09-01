#pragma once

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
