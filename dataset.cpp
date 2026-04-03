#include "dataset.h"

#include <algorithm>
#include <random>

namespace Neural {

void Dataset::ShuffleSamples(std::vector<Sample>& samples) {
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(samples.begin(), samples.end(), rng);
}

}
