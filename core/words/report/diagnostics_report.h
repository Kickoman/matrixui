#pragma once

// Console rendering for the diagnostics reports.
//
// This is the only place under core/words that is allowed to format for a
// terminal. Everything takes an explicit std::ostream&, so a GUI can either
// point it at its own terminal widget or ignore it entirely and render the
// report structs directly.

#include "core/words/diagnostics/diagnostics.h"

#include <iosfwd>

namespace Words {
class Vocabulary;
}

namespace Words::Diagnostics {

void PrintSubsamplerReport(std::ostream& out, const Vocabulary& vocabulary, const SubsamplerReport& report);
void PrintWindowSamplerReport(std::ostream& out, const Vocabulary& vocabulary, const WindowSamplerReport& report);
void PrintNegativeSamplerReport(std::ostream& out, const Vocabulary& vocabulary, const NegativeSamplerReport& report);
void PrintModelInitReport(std::ostream& out, const ModelInitReport& report);
void PrintGradientReport(std::ostream& out, const GradientReport& report);
void PrintLossBehaviourReport(std::ostream& out, const LossBehaviourReport& report);

}  // namespace Words::Diagnostics
