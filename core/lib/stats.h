#pragma once

#include <vector>

// Pearson correlation. Returns 0 rather than NaN for inputs shorter than two
// entries or when either side has zero variance.
double PearsonOf(const std::vector<double>& first, const std::vector<double>& second);

// Ranks in ascending order, starting at 1. Tied values share the average of
// the ranks they span, which is what makes PearsonOf(RanksOf(a), RanksOf(b))
// the Spearman coefficient.
std::vector<double> RanksOf(const std::vector<double>& values);
