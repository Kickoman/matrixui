#include "core/words/report/diagnostics_report.h"

#include "core/words/vocabulary.h"

#include <iomanip>
#include <ostream>

namespace Words::Diagnostics {

namespace {

const char* Verdict(const bool passed) {
    return passed ? "passed" : "FAILED";
}

}  // namespace

void PrintSubsamplerReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const SubsamplerReport& report
) {
    out << "sample t = " << report.sample << '\n';
    out << "Words affected: " << report.affectedWords << " of " << report.vocabularySize
        << "  (" << 100.0 * report.affectedWords / report.vocabularySize << "%)\n";

    out << "Corpus: " << report.corpusSize << " -> ~"
        << static_cast<std::size_t>(report.expectedCorpusLength)
        << "  (" << std::fixed << std::setprecision(1)
        << 100.0 * report.expectedCorpusLength / report.corpusSize
        << "% survives)\n\n";

    out << "Top-10 keep probabilities:\n";
    for (const auto& entry : report.topKeepProbabilities) {
        out << "  " << std::setw(8) << vocabulary.getWord(entry.id) << "  share " << std::setw(6)
            << std::setprecision(3) << 100.0 * entry.share << "%"
            << "  keep " << std::setw(6) << 100.0 * entry.keep << "%\n";
    }

    out << "\nActual after subsampling: " << report.actualCorpusLength << "  ("
        << 100.0 * report.actualCorpusLength / report.corpusSize << "%)\n";
    out << "(should closely match the estimate above)\n";

    out << "\nBefore:\n  ";
    for (const auto id : report.before) {
        out << vocabulary.getWord(id) << ' ';
    }
    out << "\n\nAfter:\n  ";
    for (const auto id : report.after) {
        out << vocabulary.getWord(id) << ' ';
    }
    out << '\n';
}

void PrintWindowSamplerReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const WindowSamplerReport& report
) {
    out << "=== Toy chunk ===\n";
    out << "chunk: ";
    for (const auto id : report.toyChunk) {
        out << vocabulary.getWord(id) << ' ';
    }
    out << "\n\n";

    auto previousCenter = std::numeric_limits<TWordId>::max();
    for (const auto& pair : report.toyPairs) {
        if (pair.center != previousCenter) {
            out << "\n  " << vocabulary.getWord(pair.center) << " -> ";
            previousCenter = pair.center;
        }
        out << vocabulary.getWord(pair.context) << ' ';
    }
    out << "\n\n";

    out << "self-pair check: " << Verdict(report.selfPair.passed) << '\n';
    out << "coverage check:  " << Verdict(report.coverage.passed) << "\n\n";

    out << "=== Full pass ===\n";
    out << "pairs generated: " << report.pairsGenerated << '\n';
    out << "expected upper bound: " << report.expectedUpperBound
        << "  (edges make the real count lower)\n\n";

    out << "Most frequent centers after subsampling:\n";
    for (const auto& [count, id] : report.topCenters) {
        out << "  " << std::setw(10) << vocabulary.getWord(id) << "  " << count << '\n';
    }
}

void PrintNegativeSamplerReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const NegativeSamplerReport& report
) {
    out << "=== Negative sampling table ===\n";
    out << "table size: " << report.tableSize << "  ("
        << report.tableSize * sizeof(TWordId) / (1024 * 1024) << " MB)\n\n";

    out << "reachable words: " << report.reachableWords << " of " << report.vocabularySize;
    out << (report.coverage.passed ? "  passed" : "  FAILED") << '\n';

    out << "\nwords tested:   " << report.wordsTested << '\n';
    out << "chi2 / df:      " << std::fixed << std::setprecision(4) << report.reducedChiSquare
        << "   (expected 1.0 +- " << report.tolerance << ")\n";
    out << (report.distribution.passed ? "  passed" : "  FAILED") << '\n';

    out << "max |z|:        " << report.maxAbsZ << " (" << vocabulary.getWord(report.worstId) << ")"
        << "   typical max ~ " << report.expectedMaxZ << '\n';

    out << "\nflattening effect (" << vocabulary.getWord(0) << " vs "
        << vocabulary.getWord(vocabulary.getSize() - 1) << "):\n";
    out << "  raw counts ratio:  " << report.rawCountRatio << '\n';
    out << "  after ^0.75:       " << report.flattenedRatio << '\n';

    out << "\nexclusion check: "
        << (report.exclusion.passed ? "passed" : "leaked (raise maxAttempts)") << '\n';
}

void PrintModelInitReport(std::ostream& out, const ModelInitReport& report) {
    out << "=== Model init ===\n";
    out << "vocab: " << report.vocabularySize << ", dim: " << report.dim << '\n';
    out << "memory: " << report.bytes / (1024 * 1024) << " MB\n\n";

    out << "output all zero:  " << Verdict(report.outputAllZero.passed) << '\n';
    out << "input mean:       " << std::scientific << std::setprecision(3) << report.inputMean
        << "  (expected ~0)\n";
    out << "input stddev:     " << report.inputStdDev
        << "  (expected " << report.expectedStdDev << ")\n";
    out << "input max |x|:    " << report.inputMaxAbs << "  (bound " << report.bound << ")\n";
    out << "bound respected:  " << Verdict(report.boundRespected.passed) << '\n';
    out << "rows distinct:    " << Verdict(report.rowsDistinct.passed) << '\n';
    out << "initial score:    " << report.initialScore << "  (must be exactly 0)\n";

    out << "\nlearning rate schedule:\n";
    for (const auto& [progress, rate] : report.learningRateSchedule) {
        out << "  " << std::fixed << std::setprecision(2) << progress << " -> "
            << std::scientific << rate << '\n';
    }
}

void PrintGradientReport(std::ostream& out, const GradientReport& report) {
    out << "=== Gradient check ===\n";
    out << std::setw(12) << "matrix" << std::setw(8) << "row" << std::setw(6) << "comp"
        << std::setw(14) << "analytic" << std::setw(14) << "numeric"
        << std::setw(12) << "rel.err" << '\n';

    for (const auto& sample : report.samples) {
        out << std::setw(12) << sample.matrix << std::setw(8) << sample.row
            << std::setw(6) << sample.component
            << std::setw(14) << std::scientific << std::setprecision(4) << sample.analytic
            << std::setw(14) << sample.numeric
            << std::setw(12) << sample.relativeError << '\n';
    }

    out << "\nworst relative error: " << report.worstRelativeError
        << (report.passed.passed ? "  passed" : "  FAILED") << '\n';
}

void PrintLossBehaviourReport(std::ostream& out, const LossBehaviourReport& report) {
    out << "\n=== Loss behaviour ===\n";
    out << "initial loss:  " << std::fixed << std::setprecision(6) << report.initialLoss
        << "   expected " << report.expectedInitialLoss
        << (report.initialLossMatches.passed ? "  passed" : "  FAILED") << '\n';

    out << "\noverfitting a single pair:\n";
    for (const auto& [step, loss] : report.lossTrace) {
        out << "  step " << std::setw(3) << step << ":  " << loss << '\n';
    }

    out << "monotone decrease: " << Verdict(report.monotoneDecrease.passed) << '\n';
}

}  // namespace Words::Diagnostics
