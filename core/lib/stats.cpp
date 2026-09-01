#include "core/lib/stats.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

double PearsonOf(const std::vector<double>& first, const std::vector<double>& second) {
    const auto count = first.size();
    if (count < 2) {
        return 0.;
    }
    const double meanFirst = std::accumulate(first.begin(), first.end(), 0.) / count;
    const double meanSecond = std::accumulate(second.begin(), second.end(), 0.) / count;

    double covariance = 0.;
    double varianceFirst = 0.;
    double varianceSecond = 0.;

    for (std::size_t i = 0; i < count; ++i) {
        const double deltaFirst = first[i] - meanFirst;
        const double deltaSecond = second[i] - meanSecond;
        covariance += deltaFirst * deltaSecond;
        varianceFirst += deltaFirst * deltaFirst;
        varianceSecond += deltaSecond * deltaSecond;
    }

    const double denominator = std::sqrt(varianceFirst * varianceSecond);
    return denominator > 0. ? covariance / denominator : 0.;
}

std::vector<double> RanksOf(const std::vector<double>& values) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](const std::size_t lhs, const std::size_t rhs) { return values[lhs] < values[rhs]; });

    std::vector<double> ranks(values.size());
    std::size_t i = 0;
    while (i < order.size()) {
        std::size_t j = i;
        while (j + 1 < order.size() && values[order[j + 1]] == values[order[i]]) {
            ++j;
        }
        const double shared = 1. * (i + j) / 2. + 1.;
        for (std::size_t k = i; k <= j; ++k) {
            ranks[order[k]] = shared;
        }
        i = j + 1;
    }
    return ranks;
}
