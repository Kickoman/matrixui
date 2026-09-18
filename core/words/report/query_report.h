#pragma once

#include "core/words/query/queries.h"

#include <iosfwd>

namespace Words {

class Vocabulary;

void PrintNeighbourReport(std::ostream& out, const Vocabulary& vocabulary, const NeighbourReport& report);
void PrintSubwordNeighbourReport(std::ostream& out, const Vocabulary& vocabulary, const SubwordNeighbourReport& report);
void PrintAnalogyQueryReport(std::ostream& out, const Vocabulary& vocabulary, const AnalogyQueryReport& report);
void PrintExpressionReport(std::ostream& out, const Vocabulary& vocabulary, const ExpressionReport& report);
void PrintOddOneOutReport(std::ostream& out, const Vocabulary& vocabulary, const OddOneOutReport& report);
void PrintAxisReport(std::ostream& out, const Vocabulary& vocabulary, const AxisReport& report);
void PrintBatteryReport(std::ostream& out, const Vocabulary& vocabulary, const BatteryReport& report);

}  // namespace Words
