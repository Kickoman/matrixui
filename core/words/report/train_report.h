#pragma once

#include "core/words/train/trainer.h"

#include <iosfwd>

namespace Words {

class Vocabulary;
class WindowSampler;

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

void PrintTrainProgress(std::ostream& out, const TrainProgress& progress);

void PrintTrainSummary(std::ostream& out, const TrainSummary& summary);

}  // namespace Words
