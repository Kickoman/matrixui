#pragma once

#include "cli/serving/options.h"

#include <iosfwd>

namespace ServingCli {

// Binds the public and the admin listener and serves until the process is
// stopped. No httplib type reaches this header: the library is included by
// server.cpp alone, which costs fourteen seconds to compile.
int RunServer(std::ostream& out, std::ostream& err, const ServeOptions& options);

}  // namespace ServingCli
