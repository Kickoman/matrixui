#include "cli/lib/argv_scan.h"

namespace CliLib {

std::string FindOptionValue(int argc, char** argv, const std::string& flag) {
    const std::string eqPrefix = flag + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == flag && i + 1 < argc) {
            return argv[i + 1];
        }
        if (arg.rfind(eqPrefix, 0) == 0) {
            return arg.substr(eqPrefix.size());
        }
    }
    return {};
}

}  // namespace CliLib
