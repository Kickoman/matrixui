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

void ModelRegistry::publish(std::shared_ptr<RegistrySnapshot> next) {
    if (!next) {
        return;
    }
    const std::lock_guard<std::mutex> guard(publishMutex);
    install(std::move(next));
}

std::shared_ptr<const RegistrySnapshot> ModelRegistry::rebuild() {
    const std::lock_guard<std::mutex> guard(publishMutex);
    install(Build(configuration));
    return slot.load();
}

void ModelRegistry::install(std::shared_ptr<RegistrySnapshot> next) {
    next->generation = ++published;
    slot.store(std::move(next));
}

}  // namespace Serving
