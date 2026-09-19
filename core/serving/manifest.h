#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace Serving {

// How a client's logical tensor flattens into the single row the network takes.
// The network itself only ever sees `size` values, so this is description, not
// instruction: nothing here is enforced on a request by the loader.
enum class InputLayout {
    Flat,
    Hwc,
};

enum class OutputKind {
    Raw,
    Classification,
    Regression,
    Embedding,
};

struct ValueRange {
    double min = 0.;
    double max = 0.;
};

struct InputContract {
    std::size_t size = 0;
    std::string dtype = "f64";
    InputLayout layout = InputLayout::Flat;
    std::vector<std::size_t> shape;
    std::optional<ValueRange> range;
};

struct OutputContract {
    std::size_t size = 0;
    OutputKind kind = OutputKind::Raw;
    std::vector<std::string> labels;
};

struct WeightsReference {
    std::filesystem::path path;
    std::string sha256;
};

struct ModelManifest {
    int manifestVersion = 0;
    std::string name;
    std::string version;
    WeightsReference weights;
    InputContract input;
    OutputContract output;
    nlohmann::json annotations = nlohmann::json::object();
};

// Parses and validates the document, resolving weights.path against
// `modelDirectory`. Throws ManifestError naming the field and the value that
// did not fit. Nothing here touches the weights file.
ModelManifest ParseManifest(const nlohmann::json& document, const std::filesystem::path& modelDirectory);

std::string_view ToString(InputLayout layout);
std::string_view ToString(OutputKind kind);

}  // namespace Serving
