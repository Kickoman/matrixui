#pragma once

#include "core/matrix/matrix.h"

#include <cstddef>
#include <filesystem>
#include <functional>

namespace CliLib {

using PngReader = std::function<Matrix(const std::filesystem::path&)>;

PngReader MakeCachedPngReader(std::size_t imageWidth, std::size_t imageHeight);

}  // namespace CliLib
