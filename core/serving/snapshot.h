#pragma once

#include "core/serving/loaded_model.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Serving {

struct ModelKey {
    std::string name;
    std::string version;
};

struct ModelKeyView {
    std::string_view name;
    std::string_view version;
};

struct ModelKeyLess {
    using is_transparent = void;

    bool operator()(const ModelKey& left, const ModelKey& right) const;
    bool operator()(const ModelKey& left, ModelKeyView right) const;
    bool operator()(ModelKeyView left, const ModelKey& right) const;
};

enum class FailureKind {
    Manifest,
    Integrity,
    Contract,
    Collision,
    Skipped,
    Budget,
};

struct ModelFailure {
    std::filesystem::path directory;
    FailureKind kind = FailureKind::Manifest;
    std::string reason;
    std::optional<ModelKey> key;
};

struct RegistrySnapshot {
    std::map<ModelKey, std::shared_ptr<const LoadedModel>, ModelKeyLess> models;
    std::map<std::string, std::string, std::less<>> defaults;
    std::vector<ModelFailure> failures;
    std::uint64_t generation = 0;
};

enum class LookupStatus {
    Found,
    UnknownModel,
    UnknownVersion,
    NoDefaultVersion,
};

struct Lookup {
    LookupStatus status = LookupStatus::UnknownModel;
    std::shared_ptr<const LoadedModel> model;
    std::vector<std::string> availableVersions;
};

Lookup Find(const RegistrySnapshot& snapshot, std::string_view name, std::string_view version);
Lookup FindDefault(const RegistrySnapshot& snapshot, std::string_view name);

std::vector<std::string> VersionsOf(const RegistrySnapshot& snapshot, std::string_view name);

struct ModelEntry {
    std::shared_ptr<const LoadedModel> model;
    bool isDefault = false;
};

struct RegistryDescription {
    std::vector<ModelEntry> models;
    std::vector<ModelFailure> failures;
};

RegistryDescription Describe(const RegistrySnapshot& snapshot);

std::string_view ToString(FailureKind kind);
std::string_view ToString(LookupStatus status);

}  // namespace Serving
