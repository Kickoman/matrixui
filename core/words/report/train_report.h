#pragma once

#include "core/words/trainer.h"

#include <iosfwd>

namespace Words {

class Vocabulary;
class WindowSampler;

// The banner printed before training starts.
void PrintTrainBanner(
    std::ostream& out,
    const ModelConfig& modelConfig,
    const SamplingConfig& samplingConfig,
    const TrainConfig& trainConfig,
    std::size_t threadCount,
    std::size_t estimatedPairs,
    std::size_t probeCount,
    double initialLoss
);

// One progress tick.
void PrintTrainProgress(std::ostream& out, const TrainProgress& progress);

// The closing summary.
void PrintTrainSummary(std::ostream& out, const TrainSummary& summary);

}  // namespace Words
