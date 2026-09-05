#include "core/words/report/inspect_report.h"

#include "core/lib/stream_format.h"

#include <ostream>

namespace Words {

void PrintCorpusStatistics(std::ostream& out, const std::string& name, const CorpusStatistics& statistics) {
    const StreamFormatGuard guard(out);

    out << "Processed " << name << '\n';
    out << "Symbols: " << statistics.symbols << '\n';
    out << "Words: " << statistics.totalWords << '\n';
    out << "Unique words: " << statistics.uniqueWords << '\n';

    for (const auto& [minCount, kept] : statistics.survivorsByMinCount) {
        out << "  words with minCount = " << minCount << ": " << kept << '\n';
    }

    out << "Top-" << statistics.topByFrequency.size() << " by frequency:\n";
    for (const auto& entry : statistics.topByFrequency) {
        out << "  " << entry.word << " of " << entry.count
            << " (share " << entry.share << ")\n";
    }
}

}  // namespace Words
