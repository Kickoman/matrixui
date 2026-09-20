#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace Serving {

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

ModelManifest ParseManifest(const nlohmann::json& document, const std::filesystem::path& modelDirectory);

std::string_view ToString(InputLayout layout);
std::string_view ToString(OutputKind kind);

}  // namespace Serving
