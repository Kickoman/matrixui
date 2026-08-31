#include "core/words/report/evaluate_report.h"

#include "core/words/report/format.h"

#include <iomanip>
#include <ostream>

namespace Words {

void PrintAnalogyReport(std::ostream& out, const AnalogyReport& report) {
    const StreamFormatGuard guard(out);

    const auto printRow = [&out](const AnalogyStats& stats) {
        out << "  " << std::setw(28) << std::left << stats.name << std::right
            << std::setw(7) << stats.asked
            << std::setw(7) << stats.skipped
            << std::setw(9) << std::fixed << std::setprecision(2)
            << 100. * stats.accuracyAdd()
            << std::setw(9) << 100. * stats.accuracyMul() << '\n';
    };

    out << "  " << std::setw(28) << std::left << "category" << std::right
        << std::setw(7) << "asked" << std::setw(7) << "skip"
        << std::setw(9) << "3CosAdd" << std::setw(9) << "3CosMul" << '\n';

    for (const auto& category : report.categories) {
        printRow(category);
    }

    out << '\n';
    printRow(report.semantic);
    printRow(report.syntactic);
    printRow(report.overall);
}

void PrintSimilarityReport(std::ostream& out, const std::string& name, const SimilarityReport& report) {
    const StreamFormatGuard guard(out);

    out << "  " << std::setw(16) << std::left << name << std::right
        << "pairs " << std::setw(5) << report.asked
        << "  skipped " << std::setw(5) << report.skipped
        << "  spearman " << std::fixed << std::setprecision(4) << report.spearman
        << "  pearson " << report.pearson << '\n';
}

}  // namespace Words
