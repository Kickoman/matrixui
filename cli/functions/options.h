#pragma once

#include "core/functions/config.h"

#include <string>
#include <string_view>

namespace FunctionsCli {

inline constexpr std::string_view kNoFunctionsToken = "none";

struct RunOptions {
    std::string dataPath;
    std::string configPath;
    std::string saveConfigPath;
    Genetizer::FunctionsConfig config{};
};

}  // namespace FunctionsCli
