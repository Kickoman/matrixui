#include "core/serving/build.h"

#include "core/serving/error.h"
#include "core/serving/model_loader.h"

#include <algorithm>
#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace Serving {

namespace {

constexpr const char* kManifestName = "manifest.json";

void Record(
    std::vector<ModelFailure>& failures,
    std::filesystem::path directory,
    const FailureKind kind,
    std::string reason,
    std::optional<ModelKey> key = std::nullopt
) {
    failures.push_back(ModelFailure{std::move(directory), kind, std::move(reason), std::move(key)});
}

std::vector<std::filesystem::path> Candidates(const RegistryConfig& config, std::vector<ModelFailure>& failures) {
    std::error_code failed;
    std::filesystem::directory_iterator entries(config.root, failed);
    if (failed) {
        throw ManifestError("Can't read the model root " + config.root.string() + ": " + failed.message());
    }

    std::vector<std::filesystem::path> candidates;
    std::size_t seen = 0;
    for (const auto& entry : entries) {
        if (++seen > config.maxRootEntries) {
            throw ManifestError(
                "The model root " + config.root.string() + " holds more than the "
                + std::to_string(config.maxRootEntries) + " entries allowed");
        }

        if (entry.is_symlink()) {
            Record(failures, entry.path(), FailureKind::Skipped,
                   "symbolic links in the model root are not followed");
            continue;
        }
        if (!std::filesystem::is_directory(entry.symlink_status())) {
            continue;
        }
        candidates.push_back(entry.path());
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

FailureKind KindOf(const Serving::Error& error) {
    if (dynamic_cast<const IntegrityError*>(&error) != nullptr) {
        return FailureKind::Integrity;
    }
    if (dynamic_cast<const ContractError*>(&error) != nullptr) {
        return FailureKind::Contract;
    }
    return FailureKind::Manifest;
}

std::uint64_t DeclaredWeightBytes(const Neural::NeuralNetwork& network) {
    std::uint64_t bytes = 0;
    const auto& sizes = network.config.layersSizes;
    for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
        const std::uint64_t rows = sizes[i];
        const std::uint64_t cols = sizes[i + 1];
        bytes += (rows * cols + cols) * sizeof(double);
    }
    return bytes;
}

}  // namespace

std::shared_ptr<RegistrySnapshot> Build(const RegistryConfig& config) {
    std::error_code ignored;
    if (!std::filesystem::is_directory(config.root, ignored)) {
        throw ManifestError("Not a model root: " + config.root.string());
    }

    auto snapshot = std::make_shared<RegistrySnapshot>();
    const auto candidates = Candidates(config, snapshot->failures);

    struct Claim {
        std::filesystem::path directory;
        std::uint64_t declaredBytes = 0;
    };
    std::map<ModelKey, Claim, ModelKeyLess> claimedBy;
    std::map<ModelKey, bool, ModelKeyLess> collided;
    std::uint64_t declaredBytes = 0;

    for (const auto& directory : candidates) {
        if (!std::filesystem::is_regular_file(directory / kManifestName, ignored)) {
            Record(snapshot->failures, directory, FailureKind::Skipped,
                   "no " + std::string(kManifestName));
            continue;
        }

        std::optional<LoadedModel> loaded;
        try {
            loaded.emplace(LoadModel(directory));
        } catch (const Serving::Error& error) {
            Record(snapshot->failures, directory, KindOf(error), error.what());
            continue;
        }

        const ModelKey key{loaded->manifest().name, loaded->manifest().version};

        if (const auto claimed = claimedBy.find(key); claimed != claimedBy.end()) {
            const auto reason =
                "model \"" + key.name + "\" version \"" + key.version
                + "\" is declared by two directories:\n  " + claimed->second.directory.string()
                + "\n  " + directory.string();
            Record(snapshot->failures, claimed->second.directory, FailureKind::Collision, reason, key);
            declaredBytes -= claimed->second.declaredBytes;
            snapshot->models.erase(key);
            claimedBy.erase(claimed);
            collided.emplace(key, true);
            Record(snapshot->failures, directory, FailureKind::Collision, reason, key);
            continue;
        }
        if (collided.find(key) != collided.end()) {
            Record(snapshot->failures, directory, FailureKind::Collision,
                   "model \"" + key.name + "\" version \"" + key.version
                   + "\" is declared by more than one directory", key);
            continue;
        }

        const auto bytes = DeclaredWeightBytes(*loaded->network());
        if (declaredBytes + bytes > config.maxDeclaredWeightBytes) {
            Record(snapshot->failures, directory, FailureKind::Budget,
                   "loading this model would declare " + std::to_string(declaredBytes + bytes)
                   + " bytes of weights, more than the "
                   + std::to_string(config.maxDeclaredWeightBytes) + " allowed",
                   key);
            continue;
        }
        declaredBytes += bytes;

        claimedBy.emplace(key, Claim{directory, bytes});
        snapshot->models.emplace(key, std::make_shared<const LoadedModel>(std::move(*loaded)));
    }

    for (const auto& [name, version] : config.defaults) {
        if (snapshot->models.find(ModelKeyView{name, version}) != snapshot->models.end()) {
            snapshot->defaults.emplace(name, version);
            continue;
        }
        auto available = VersionsOf(*snapshot, name);
        std::string reason = "default version \"" + version + "\" for model \"" + name + "\" is not loaded";
        if (available.empty()) {
            reason += " (no versions of that model are loaded)";
        } else {
            reason += " (loaded: ";
            for (std::size_t i = 0; i < available.size(); ++i) {
                reason += (i == 0 ? "" : ", ") + available[i];
            }
            reason += ")";
        }
        Record(snapshot->failures, config.root, FailureKind::Manifest, reason, ModelKey{name, version});
    }

    std::sort(snapshot->failures.begin(), snapshot->failures.end(),
              [](const ModelFailure& left, const ModelFailure& right) {
                  return left.directory < right.directory;
              });
    return snapshot;
}

}  // namespace Serving
