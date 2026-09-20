#include "core/serving/registry.h"

#include "core/serving/build.h"

#include <utility>

namespace Serving {

ModelRegistry::ModelRegistry(RegistryConfig config)
    : slot(std::make_shared<const RegistrySnapshot>())
    , configuration(std::move(config))
{ }

std::shared_ptr<const RegistrySnapshot> ModelRegistry::snapshot() const {
    return slot.load();
}

void ModelRegistry::publish(RegistrySnapshot&& next) {
    const std::lock_guard<std::mutex> guard(publishMutex);
    install(std::move(next));
}

std::shared_ptr<const RegistrySnapshot> ModelRegistry::rebuild() {
    const std::lock_guard<std::mutex> guard(publishMutex);
    install(Build(configuration));
    return slot.load();
}

void ModelRegistry::install(RegistrySnapshot&& next) {
    next.generation = ++published;
    slot.store(std::make_shared<const RegistrySnapshot>(std::move(next)));
}

}  // namespace Serving
