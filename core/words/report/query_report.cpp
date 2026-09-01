#include "core/words/report/query_report.h"

#include "core/lib/stream_format.h"
#include "core/words/data/vocabulary.h"

#include <iomanip>
#include <ostream>

namespace Words {

namespace {

template <typename Rows, typename Score>
void PrintScoreRows(std::ostream& out, const Vocabulary& vocabulary, const Rows& rows, Score score) {
    for (const auto& row : rows) {
        out << "    " << std::setw(18) << std::left << vocabulary.getWord(row.id)
            << std::right << std::fixed << std::setprecision(4) << score(row) << '\n';
    }
}

double SimilarityOf(const Neighbour& neighbour) { return neighbour.similarity; }
double ScoreOf(const ScoredWord& word) { return word.score; }

void PrintSignedRows(std::ostream& out, const Vocabulary& vocabulary, const std::vector<ScoredWord>& rows) {
    for (const auto& row : rows) {
        out << "  " << std::setw(6) << std::right << std::showpos << std::fixed
            << std::setprecision(4) << row.score << std::noshowpos
            << "  " << vocabulary.getWord(row.id) << '\n';
    }
}

}  // namespace

void PrintNeighbourReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const NeighbourReport& report
) {
    const StreamFormatGuard guard(out);

    if (!report.status.ok) {
        out << "  " << report.status.message << '\n';
        return;
    }

    out << report.word << " (id " << report.id
        << ", count " << vocabulary.getCount(report.id) << "):\n";
    PrintScoreRows(out, vocabulary, report.neighbours, SimilarityOf);
    out << '\n';
}

void PrintAnalogyQueryReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const AnalogyQueryReport& report
) {
    const StreamFormatGuard guard(out);

    if (!report.status.ok) {
        out << "  " << report.status.message << "\n\n";
        return;
    }

    out << report.b << " - " << report.a << " + " << report.c << ":\n";
    PrintScoreRows(out, vocabulary, report.neighbours, SimilarityOf);
    out << '\n';
}

void PrintExpressionReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const ExpressionReport& report
) {
    const StreamFormatGuard guard(out);

    if (!report.status.ok) {
        out << "  " << report.status.message << '\n';
        return;
    }

    out << report.expression << "\n\n";
    out << "  3CosAdd:\n";
    PrintScoreRows(out, vocabulary, report.cosAdd, SimilarityOf);

    if (!report.analogyShape) {
        return;
    }

    out << "\n  3CosMul:\n";
    PrintScoreRows(out, vocabulary, report.cosMul, ScoreOf);
}

void PrintOddOneOutReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const OddOneOutReport& report
) {
    const StreamFormatGuard guard(out);

    if (!report.status.ok) {
        out << "  " << report.status.message << '\n';
        return;
    }

    out << "  similarity to the centroid:\n";
    PrintScoreRows(out, vocabulary, report.scored, ScoreOf);
    out << "\n  odd one out: " << vocabulary.getWord(report.oddOne) << '\n';
}

void PrintAxisReport(std::ostream& out, const Vocabulary& vocabulary, const AxisReport& report) {
    const StreamFormatGuard guard(out);

    if (!report.status.ok) {
        out << "  " << report.status.message << '\n';
        return;
    }

    out << "axis: " << report.axis << "\n\n";

    if (report.explicitWordList) {
        PrintSignedRows(out, vocabulary, report.ranked);
        return;
    }

    out << "  positive end:\n";
    PrintSignedRows(out, vocabulary, report.positive);
    out << "\n  negative end:\n";
    PrintSignedRows(out, vocabulary, report.negative);
}

void PrintBatteryReport(
    std::ostream& out,
    const Vocabulary& vocabulary,
    const BatteryReport& report
) {
    out << "=== Nearest neighbours ===\n\n";
    for (const auto& neighbours : report.neighbours) {
        PrintNeighbourReport(out, vocabulary, neighbours);
    }

    out << "=== Analogies ===\n\n";
    for (const auto& analogy : report.analogies) {
        PrintAnalogyQueryReport(out, vocabulary, analogy);
    }
}

}  // namespace Words
