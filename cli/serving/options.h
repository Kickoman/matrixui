#pragma once

#include <map>
#include <string>

namespace ServingCli {

struct ListOptions {
    std::string root;
    std::map<std::string, std::string> defaults;
};

}  // namespace ServingCli
