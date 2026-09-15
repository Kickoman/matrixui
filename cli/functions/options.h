#pragma once

#include "core/functions/config.h"

#include <string>

namespace FunctionsCli {

struct RunOptions {
    std::string dataPath;
    std::string configPath;
    std::string saveConfigPath;
    Genetizer::FunctionsConfig config{};
};

}  // namespace FunctionsCli
