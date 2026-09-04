#pragma once

#include <string>

namespace CliLib {

std::string FindOptionValue(int argc, char** argv, const std::string& flag);

}  // namespace CliLib
