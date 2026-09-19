#include "core/serving/manifest.h"

#include "core/serving/error.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace Serving {

namespace {

constexpr int kSupportedManifestVersion = 1;
constexpr std::size_t kMaxIdentifier = 64;
constexpr std::size_t kSha256HexDigits = 64;

[[noreturn]] void Refuse(const std::string& message) {
    throw ManifestError("manifest.json: " + message);
}

std::string Describe(const nlohmann::json& value) {
    return value.is_string() ? "\"" + value.get<std::string>() + "\"" : value.dump();
}

const nlohmann::json* Field(const nlohmann::json& object, const std::string& where, const std::string& key) {
    const auto found = object.find(key);
    if (found == object.end()) {
        Refuse("missing required key \"" + key + "\" in " + where);
    }
    return &*found;
}

void RejectUnknownKeys(
    const nlohmann::json& object,
    const std::string& where,
    const std::vector<std::string>& known
) {
    for (const auto& entry : object.items()) {
        if (std::find(known.begin(), known.end(), entry.key()) == known.end()) {
            std::string list;
            for (const auto& key : known) {
                list += (list.empty() ? "" : ", ") + key;
            }
            Refuse("unknown key \"" + entry.key() + "\" in " + where + " (known: " + list + ")");
        }
    }
}

void RequireObject(const nlohmann::json& value, const std::string& where) {
    if (!value.is_object()) {
        Refuse(where + " must be an object, found " + Describe(value));
    }
}

std::size_t AsPositiveSize(const nlohmann::json& value, const std::string& where) {
    bool positive = false;
    if (value.is_number_unsigned()) {
        positive = value.get<std::uint64_t>() > 0;
    } else if (value.is_number_integer()) {
        positive = value.get<std::int64_t>() > 0;
    }
    if (!positive) {
        Refuse(where + " must be a positive integer, found " + Describe(value));
    }
    return value.get<std::size_t>();
}

std::string AsString(const nlohmann::json& value, const std::string& where) {
    if (!value.is_string()) {
        Refuse(where + " must be a string, found " + Describe(value));
    }
    return value.get<std::string>();
}

std::string AsIdentifier(const nlohmann::json& value, const std::string& where) {
    const std::string text = AsString(value, where);
    const auto isTail = [](const unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    };
    const auto isHead = [](const unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
    };
    const bool valid = !text.empty()
        && text.size() <= kMaxIdentifier
        && isHead(static_cast<unsigned char>(text.front()))
        && std::all_of(text.begin() + 1, text.end(), [&](const char c) {
               return isTail(static_cast<unsigned char>(c));
           });
    if (!valid) {
        Refuse(where + " \"" + text + "\" does not match [a-z0-9][a-z0-9_-]{0,63}");
    }
    return text;
}

bool IsInside(const std::filesystem::path& directory, const std::filesystem::path& candidate) {
    auto expected = directory.begin();
    auto actual = candidate.begin();
    for (; expected != directory.end(); ++expected, ++actual) {
        if (actual == candidate.end() || *actual != *expected) {
            return false;
        }
    }
    return true;
}

WeightsReference ParseWeights(const nlohmann::json& object, const std::filesystem::path& modelDirectory) {
    RequireObject(object, "weights");
    RejectUnknownKeys(object, "weights", {"path", "sha256"});

    const std::string declared = AsString(*Field(object, "weights", "path"), "weights.path");
    const std::filesystem::path relative(declared);
    if (declared.empty() || relative.is_absolute()) {
        Refuse("weights.path \"" + declared + "\" must be relative to the model directory");
    }

    const auto root = std::filesystem::weakly_canonical(modelDirectory);
    const auto resolved = std::filesystem::weakly_canonical(modelDirectory / relative);
    if (!IsInside(root, resolved)) {
        Refuse("weights.path \"" + declared + "\" escapes the model directory");
    }

    WeightsReference weights;
    weights.path = resolved;
    if (const auto found = object.find("sha256"); found != object.end()) {
        weights.sha256 = AsString(*found, "weights.sha256");
        const bool wellFormed = weights.sha256.size() == kSha256HexDigits
            && std::all_of(weights.sha256.begin(), weights.sha256.end(), [](const unsigned char c) {
                   return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
               });
        if (!wellFormed) {
            Refuse("weights.sha256 must be 64 lowercase hex digits, found \"" + weights.sha256 + "\"");
        }
    }
    return weights;
}

InputContract ParseInput(const nlohmann::json& object) {
    RequireObject(object, "input");
    RejectUnknownKeys(object, "input", {"size", "dtype", "layout", "shape", "range"});

    InputContract input;
    input.size = AsPositiveSize(*Field(object, "input", "size"), "input.size");

    if (const auto found = object.find("dtype"); found != object.end()) {
        input.dtype = AsString(*found, "input.dtype");
        // Matrix holds Eigen::MatrixXd, so nothing else can be served yet.
        if (input.dtype != "f64") {
            Refuse("input.dtype \"" + input.dtype + "\" is not supported (only f64)");
        }
    }

    if (const auto found = object.find("layout"); found != object.end()) {
        const std::string layout = AsString(*found, "input.layout");
        if (layout == "flat") {
            input.layout = InputLayout::Flat;
        } else if (layout == "hwc") {
            input.layout = InputLayout::Hwc;
        } else {
            Refuse("input.layout \"" + layout + "\" is not one of flat, hwc");
        }
    }

    if (const auto found = object.find("shape"); found != object.end()) {
        if (!found->is_array() || found->empty()) {
            Refuse("input.shape must be a non-empty array, found " + Describe(*found));
        }
        std::size_t product = 1;
        for (std::size_t i = 0; i < found->size(); ++i) {
            const auto dimension = AsPositiveSize((*found)[i], "input.shape[" + std::to_string(i) + "]");
            input.shape.push_back(dimension);
            product *= dimension;
        }
        if (product != input.size) {
            Refuse("input.shape " + found->dump() + " multiplies out to " + std::to_string(product)
                   + ", but input.size is " + std::to_string(input.size));
        }
    }

    if (const auto found = object.find("range"); found != object.end()) {
        const nlohmann::json& range = *found;
        RequireObject(range, "input.range");
        RejectUnknownKeys(range, "input.range", {"min", "max"});
        const nlohmann::json& min = *Field(range, "input.range", "min");
        const nlohmann::json& max = *Field(range, "input.range", "max");
        if (!min.is_number() || !max.is_number()) {
            Refuse("input.range.min and input.range.max must be numbers");
        }
        ValueRange bounds{min.get<double>(), max.get<double>()};
        if (bounds.min > bounds.max) {
            Refuse("input.range has min " + min.dump() + " above max " + max.dump());
        }
        input.range = bounds;
    }

    return input;
}

OutputContract ParseOutput(const nlohmann::json& object) {
    RequireObject(object, "output");
    RejectUnknownKeys(object, "output", {"size", "kind", "labels"});

    OutputContract output;
    output.size = AsPositiveSize(*Field(object, "output", "size"), "output.size");

    if (const auto found = object.find("kind"); found != object.end()) {
        const std::string kind = AsString(*found, "output.kind");
        if (kind == "raw") {
            output.kind = OutputKind::Raw;
        } else if (kind == "classification") {
            output.kind = OutputKind::Classification;
        } else if (kind == "regression") {
            output.kind = OutputKind::Regression;
        } else if (kind == "embedding") {
            output.kind = OutputKind::Embedding;
        } else {
            Refuse("output.kind \"" + kind + "\" is not one of raw, classification, regression, embedding");
        }
    }

    if (const auto found = object.find("labels"); found != object.end()) {
        if (!found->is_array()) {
            Refuse("output.labels must be an array, found " + Describe(*found));
        }
        for (std::size_t i = 0; i < found->size(); ++i) {
            output.labels.push_back(AsString((*found)[i], "output.labels[" + std::to_string(i) + "]"));
        }
        if (output.labels.size() != output.size) {
            Refuse("output.labels has " + std::to_string(output.labels.size())
                   + " entries, but output.size is " + std::to_string(output.size));
        }
    }

    return output;
}

}  // namespace

std::string_view ToString(const InputLayout layout) {
    switch (layout) {
        case InputLayout::Flat: return "flat";
        case InputLayout::Hwc:  return "hwc";
    }
    return "?";
}

std::string_view ToString(const OutputKind kind) {
    switch (kind) {
        case OutputKind::Raw:            return "raw";
        case OutputKind::Classification: return "classification";
        case OutputKind::Regression:     return "regression";
        case OutputKind::Embedding:      return "embedding";
    }
    return "?";
}

ModelManifest ParseManifest(const nlohmann::json& document, const std::filesystem::path& modelDirectory) {
    RequireObject(document, "root");
    const nlohmann::json& root = document;
    RejectUnknownKeys(root, "root",
        {"manifestVersion", "name", "version", "weights", "input", "output", "annotations"});

    const nlohmann::json& declaredVersion = *Field(root, "root", "manifestVersion");
    if (!declaredVersion.is_number_integer()) {
        Refuse("manifestVersion must be an integer, found " + Describe(declaredVersion));
    }

    ModelManifest manifest;
    manifest.manifestVersion = declaredVersion.get<int>();
    if (manifest.manifestVersion != kSupportedManifestVersion) {
        Refuse("unsupported manifestVersion " + std::to_string(manifest.manifestVersion)
               + " (supported: " + std::to_string(kSupportedManifestVersion) + ")");
    }

    manifest.name = AsIdentifier(*Field(root, "root", "name"), "name");
    manifest.version = AsIdentifier(*Field(root, "root", "version"), "version");
    manifest.weights = ParseWeights(*Field(root, "root", "weights"), modelDirectory);
    manifest.input = ParseInput(*Field(root, "root", "input"));
    manifest.output = ParseOutput(*Field(root, "root", "output"));

    if (const auto found = root.find("annotations"); found != root.end()) {
        RequireObject(*found, "annotations");
        manifest.annotations = *found;
    }

    return manifest;
}

}  // namespace Serving
