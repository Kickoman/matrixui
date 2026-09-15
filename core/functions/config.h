#pragma once

#include "core/functions/applier.h"
#include "core/functions/genetizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Genetizer {

struct MutationOptions {
    std::string operators{"+-*/^"};
    double scalarRange{5.0};
};

struct FunctionsConfig {
    genetyka::GenetizerConfig genetizer{};
    MutationOptions mutation{};
    FitnessConfig fitness{};
    std::vector<std::string> initialExpressions{};
    std::size_t randomCount{200};
    std::size_t randomDepth{4};
    std::size_t epochs{500};
    std::size_t patience{50};
    std::size_t printTop{10};
    std::size_t printEvery{10};
    std::uint64_t seed{0};
};

}  // namespace Genetizer
