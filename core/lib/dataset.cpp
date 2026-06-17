#include "core/lib/dataset.h"

#include <algorithm>
#include <random>

namespace Neural {

void Dataset::ShuffleSamples(std::vector<Sample>& samples) {
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(samples.begin(), samples.end(), rng);
}

void Dataset::FilterSamples(std::vector<Sample> &samples, const std::size_t labelsCount) {
    auto it = std::remove_if(samples.begin(), samples.end(), [labelsCount](const Sample& sample) -> bool {
        return sample.label >= labelsCount;
    });
    samples.erase(it, samples.end());
}

}
