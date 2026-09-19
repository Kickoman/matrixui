#pragma once

#include "core/serving/loaded_model.h"

#include <filesystem>

namespace Serving {

// Turns a model directory -- manifest.json plus the weights blob it names --
// into a LoadedModel, or refuses saying which number did not match.
//
// Throws ManifestError, IntegrityError or ContractError. Knows nothing about
// image formats: the manifest describes the input contract, this does not
// enforce it.
LoadedModel LoadModel(const std::filesystem::path& directory);

}  // namespace Serving
