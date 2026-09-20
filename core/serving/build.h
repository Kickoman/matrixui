#pragma once

#include "core/serving/registry.h"
#include "core/serving/snapshot.h"

namespace Serving {

RegistrySnapshot Build(const RegistryConfig& config);

}  // namespace Serving
