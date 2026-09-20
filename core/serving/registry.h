#pragma once

#include "core/serving/snapshot.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace Serving {

struct RegistryConfig {
    std::filesystem::path root;

    std::map<std::string, std::string> defaults;

    std::size_t maxRootEntries = 4096;
    std::uint64_t maxDeclaredWeightBytes = 2ull << 30;
};

class ModelRegistry {
public:
    explicit ModelRegistry(RegistryConfig config);

    std::shared_ptr<const RegistrySnapshot> snapshot() const;

    void publish(RegistrySnapshot&& next);

    std::shared_ptr<const RegistrySnapshot> rebuild();

    const RegistryConfig& config() const { return configuration; }

private:
    void install(RegistrySnapshot&& next);

    alignas(64) std::atomic<std::shared_ptr<const RegistrySnapshot>> slot;
    alignas(64) RegistryConfig configuration;

    mutable std::mutex publishMutex;
    std::uint64_t published = 0;
};

}  // namespace Serving
