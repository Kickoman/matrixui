#pragma once

#include <nlohmann/json.hpp>

#include <exception>
#include <fstream>
#include <ostream>
#include <string>

namespace CliLib {

template <typename Config>
bool LoadJsonConfig(std::ostream& err, const std::string& path, Config& config, const char* label) {
    if (path.empty()) {
        return true;
    }
    std::ifstream in(path);
    if (!in) {
        err << "Failed to open " << label << " config file: " << path << "\n";
        return false;
    }
    try {
        nlohmann::json j;
        in >> j;
        config = j.get<Config>();
    } catch (const std::exception& e) {
        err << "Failed to parse " << label << " config (" << path << "): " << e.what() << "\n";
        return false;
    }
    return true;
}

}  // namespace CliLib
