#pragma once

#include "core/serving/loaded_model.h"

#include <filesystem>

namespace Serving {

LoadedModel LoadModel(const std::filesystem::path& directory);

}  // namespace Serving
