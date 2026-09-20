#pragma once

#include "core/serving/registry.h"
#include "core/serving/snapshot.h"

#include <memory>

namespace Serving {

std::shared_ptr<RegistrySnapshot> Build(const RegistryConfig& config);

}  // namespace Serving
