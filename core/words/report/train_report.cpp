#include "core/words/report/train_report.h"

#include "core/words/report/format.h"

#include <iomanip>
#include <ostream>

namespace Words {

void PrintTrainBanner(
    std::ostream& out,
    const ModelConfig& modelConfig,
    const SamplingConfig& samplingConfig,
    const TrainConfig& trainConfig,
    const std::size_t threadCount,
    const std::size_t estimatedPairs,
    const std::size_t probeCount,
    const double initialLoss
) {
    const StreamFormatGuard guard(out);

    out << "dim " << modelConfig.dim
        << ", negatives " << modelConfig.negatives
        << ", window " << samplingConfig.window
        << ", epochs " << trainConfig.epochs
        << ", threads " << threadCount << '\n';
    out << "estimated pairs: " << estimatedPairs << '\n';
    out << "probe set: " << probeCount << " pairs\n";
    out << "initial loss: " << std::fixed << std::setprecision(4) << initialLoss << "\n\n";
}

void PrintTrainProgress(std::ostream& out, const TrainProgress& progress) {
    const StreamFormatGuard guard(out);

    out << std::fixed << std::setprecision(1)
        << std::setw(5) << 100. * progress.progress << "%"
        << "  pairs " << std::setw(12) << progress.pairsDone
        << "  lr " << std::scientific << std::setprecision(3) << progress.learningRate
        << "  loss " << std::fixed << std::setprecision(4) << progress.loss
        << "  " << std::setprecision(0) << progress.pairsPerSecond / 1000. << "k pairs/s"
        << "  eta " << progress.etaSeconds << "s"
        << "  elapsed " << progress.elapsedSeconds << "s\n";
}

void PrintTrainSummary(std::ostream& out, const TrainSummary& summary) {
    const StreamFormatGuard guard(out);

    out << (summary.stopped ? "\nstopped: " : "\ndone: ") << summary.pairsDone
        << " pairs in " << std::fixed << std::setprecision(1) << summary.elapsedSeconds
        << "s (" << summary.pairsPerSecond / 1000. << "k pairs/s)\n";
    out << "estimate was " << summary.pairsEstimated << ", ratio "
        << std::setprecision(3)
        << (summary.pairsEstimated > 0
                ? static_cast<double>(summary.pairsDone) / summary.pairsEstimated
                : 0.)
        << '\n';
    out << "final loss: " << std::setprecision(4) << summary.finalLoss << '\n';
}

}  // namespace Words
