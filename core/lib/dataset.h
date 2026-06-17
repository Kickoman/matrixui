#pragma once

#include "matrix/matrix.h"

#include <vector>

namespace Neural {

struct Sample {
    Matrix input;
    std::size_t label;
};


class Dataset {
public:
    virtual ~Dataset() = default;

    static void FilterSamples(std::vector<Sample>& samples, std::size_t labelsCount);
    static void ShuffleSamples(std::vector<Sample>& samples);

    virtual std::vector<Sample> getSamplesForLabel(const std::size_t label, const std::size_t limit = 0) const = 0;
    virtual std::vector<Sample> getAllSamples(std::size_t limitPerLabel = 0) const = 0;
};

}
