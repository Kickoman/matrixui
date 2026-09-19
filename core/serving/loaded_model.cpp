#include "core/serving/loaded_model.h"

namespace Serving {

LoadedModel::LoadedModel(
    ModelManifest manifest,
    std::shared_ptr<const Neural::NeuralNetwork> network,
    std::filesystem::path directory,
    const IntegrityCheck integrity
)
    : manifestData(std::move(manifest))
    , networkData(std::move(network))
    , directoryPath(std::move(directory))
    , integrityState(integrity)
{ }

}  // namespace Serving
