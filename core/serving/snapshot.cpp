#include "core/serving/snapshot.h"

#include <string_view>
#include <utility>

namespace Serving {

namespace {

bool KeyLess(
    const std::string_view leftName, const std::string_view leftVersion,
    const std::string_view rightName, const std::string_view rightVersion
) {
    if (leftName != rightName) {
        return leftName < rightName;
    }
    return leftVersion < rightVersion;
}

}  // namespace

bool ModelKeyLess::operator()(const ModelKey& left, const ModelKey& right) const {
    return KeyLess(left.name, left.version, right.name, right.version);
}

bool ModelKeyLess::operator()(const ModelKey& left, const ModelKeyView right) const {
    return KeyLess(left.name, left.version, right.name, right.version);
}

bool ModelKeyLess::operator()(const ModelKeyView left, const ModelKey& right) const {
    return KeyLess(left.name, left.version, right.name, right.version);
}

std::vector<std::string> VersionsOf(const RegistrySnapshot& snapshot, const std::string_view name) {
    std::vector<std::string> versions;
    for (auto it = snapshot.models.lower_bound(ModelKeyView{name, std::string_view{}});
         it != snapshot.models.end() && it->first.name == name;
         ++it) {
        versions.push_back(it->first.version);
    }
    return versions;
}

Lookup Find(const RegistrySnapshot& snapshot, const std::string_view name, const std::string_view version) {
    if (const auto found = snapshot.models.find(ModelKeyView{name, version});
        found != snapshot.models.end()) {
        return Lookup{LookupStatus::Found, found->second, {}};
    }

    auto versions = VersionsOf(snapshot, name);
    if (versions.empty()) {
        return Lookup{LookupStatus::UnknownModel, nullptr, {}};
    }
    return Lookup{LookupStatus::UnknownVersion, nullptr, std::move(versions)};
}

Lookup FindDefault(const RegistrySnapshot& snapshot, const std::string_view name) {
    if (const auto chosen = snapshot.defaults.find(name); chosen != snapshot.defaults.end()) {
        return Find(snapshot, name, chosen->second);
    }

    auto versions = VersionsOf(snapshot, name);
    if (versions.empty()) {
        return Lookup{LookupStatus::UnknownModel, nullptr, {}};
    }
    return Lookup{LookupStatus::NoDefaultVersion, nullptr, std::move(versions)};
}

RegistryDescription Describe(const RegistrySnapshot& snapshot) {
    RegistryDescription description;
    description.models.reserve(snapshot.models.size());
    for (const auto& [key, model] : snapshot.models) {
        const auto chosen = snapshot.defaults.find(key.name);
        description.models.push_back(ModelEntry{
            model,
            chosen != snapshot.defaults.end() && chosen->second == key.version,
        });
    }
    description.failures = snapshot.failures;
    return description;
}

std::string_view ToString(const FailureKind kind) {
    switch (kind) {
        case FailureKind::Manifest:  return "manifest";
        case FailureKind::Integrity: return "integrity";
        case FailureKind::Contract:  return "contract";
        case FailureKind::Collision: return "collision";
        case FailureKind::Skipped:   return "skipped";
        case FailureKind::Budget:    return "budget";
    }
    return "?";
}

std::string_view ToString(const LookupStatus status) {
    switch (status) {
        case LookupStatus::Found:            return "found";
        case LookupStatus::UnknownModel:     return "unknown model";
        case LookupStatus::UnknownVersion:   return "unknown version";
        case LookupStatus::NoDefaultVersion: return "no default version";
    }
    return "?";
}

}  // namespace Serving
