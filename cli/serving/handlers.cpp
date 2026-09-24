#include "cli/serving/handlers.h"

#include "core/lib/write.h"
#include "core/matrix/matrix.h"
#include "core/nn/neural_network_applier.h"
#include "core/serving/loaded_model.h"
#include "core/serving/manifest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ServingCli {

namespace {

constexpr char kJson[] = "application/json";
constexpr char kBinary[] = "application/octet-stream";
constexpr std::size_t kMaxParseMessageBytes = 512;

// The wire is little-endian whatever the host is, the same rule the on-disk
// formats follow. IsLittleEndian() is constexpr, so the swap disappears on x86.
double ReadLittleEndianDouble(const char* at) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, at, sizeof(bits));
    if constexpr (!IsLittleEndian()) {
        bits = SwapBytes(bits);
    }
    double value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void WriteLittleEndianDouble(char* at, const double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if constexpr (!IsLittleEndian()) {
        bits = SwapBytes(bits);
    }
    std::memcpy(at, &bits, sizeof(bits));
}

enum class BodyFormat {
    Json,
    Binary,
    Unsupported,
};

BodyFormat FormatOf(std::string_view contentType) {
    const auto semicolon = contentType.find(';');
    std::string_view type = contentType.substr(0, semicolon);
    while (!type.empty() && (type.back() == ' ' || type.back() == '\t')) {
        type.remove_suffix(1);
    }
    if (type == kJson) {
        return BodyFormat::Json;
    }
    if (type == kBinary) {
        return BodyFormat::Binary;
    }
    return BodyFormat::Unsupported;
}

HttpReply Reply(const int status, nlohmann::json body) {
    return HttpReply{status, kJson, body.dump() + "\n", {}};
}

std::string Truncated(const std::string& text) {
    if (text.size() <= kMaxParseMessageBytes) {
        return text;
    }
    return text.substr(0, kMaxParseMessageBytes) + "...";
}

nlohmann::json ManifestJson(const Serving::LoadedModel& model, const bool isDefault) {
    const auto& manifest = model.manifest();

    nlohmann::json input{
        {"size", manifest.input.size},
        {"dtype", manifest.input.dtype},
        {"layout", Serving::ToString(manifest.input.layout)},
    };
    if (!manifest.input.shape.empty()) {
        input["shape"] = manifest.input.shape;
    }
    if (manifest.input.range.has_value()) {
        input["range"] = nlohmann::json{
            {"min", manifest.input.range->min},
            {"max", manifest.input.range->max},
        };
    }

    nlohmann::json output{
        {"size", manifest.output.size},
        {"kind", Serving::ToString(manifest.output.kind)},
    };
    if (!manifest.output.labels.empty()) {
        output["labels"] = manifest.output.labels;
    }

    return nlohmann::json{
        {"name", manifest.name},
        {"version", manifest.version},
        {"isDefault", isDefault},
        {"integrity", model.integrity() == Serving::IntegrityCheck::Verified ? "verified" : "not declared"},
        {"input", std::move(input)},
        {"output", std::move(output)},
    };
}

bool IsDefault(const Serving::RegistrySnapshot& snapshot, const Serving::ModelKey& key) {
    const auto at = snapshot.defaults.find(key.name);
    return at != snapshot.defaults.end() && at->second == key.version;
}

// The registry answers a miss with data, so this is the one place a miss becomes
// a status. NoDefaultVersion cannot reach a versioned route, but the switch
// stays exhaustive so a later change cannot fall through to an empty body.
HttpReply LookupFailure(const Serving::Lookup& lookup, std::string_view name, std::string_view version) {
    switch (lookup.status) {
        case Serving::LookupStatus::UnknownModel:
            return Reply(404, {{"error", {
                {"code", "unknown_model"},
                {"message", "no model named \"" + std::string(name) + "\""},
            }}});
        case Serving::LookupStatus::UnknownVersion:
            return Reply(404, {{"error", {
                {"code", "unknown_version"},
                {"message", "model \"" + std::string(name) + "\" has no version \"" + std::string(version) + "\""},
                {"availableVersions", lookup.availableVersions},
            }}});
        case Serving::LookupStatus::NoDefaultVersion:
            return Reply(400, {{"error", {
                {"code", "no_default_version"},
                {"message", "model \"" + std::string(name) + "\" has no default version; name one"},
                {"availableVersions", lookup.availableVersions},
            }}});
        case Serving::LookupStatus::Found:
            break;
    }
    return Reply(500, {{"error", {{"code", "internal"}, {"message", "internal error"}}}});
}

struct Decoded {
    std::size_t rows = 0;
    std::vector<double> values;
};

HttpReply BadInputSize(const std::string& message) {
    return Reply(400, {{"error", {{"code", "bad_input_size"}, {"message", message}}}});
}

HttpReply BatchTooLarge(const std::size_t rows, const std::size_t allowed) {
    return Reply(413, {{"error", {
        {"code", "batch_too_large"},
        {"message", std::to_string(rows) + " rows, more than the "
                    + std::to_string(allowed) + " allowed"},
    }}});
}

HttpReply MalformedBody(const std::string& message) {
    return Reply(400, {{"error", {{"code", "malformed_body"}, {"message", message}}}});
}

std::string ModelName(const Serving::ModelManifest& manifest) {
    return "model \"" + manifest.name + "\" " + manifest.version;
}

// Binary carries no width of its own: the manifest is the only thing that says
// how the bytes divide into rows.
bool DecodeBinary(
    std::string_view body,
    const Serving::ModelManifest& manifest,
    Decoded& into,
    HttpReply& failure
) {
    // No ceiling needed before allocating here: rows is derived from body.size(),
    // which httplib has already bounded by maxBodyBytes.

    const std::size_t rowBytes = manifest.input.size * sizeof(double);
    if (body.size() % rowBytes != 0) {
        failure = BadInputSize(
            ModelName(manifest) + " takes " + std::to_string(manifest.input.size)
            + " values per row, so a body must be a multiple of " + std::to_string(rowBytes)
            + " bytes; got " + std::to_string(body.size())
        );
        return false;
    }
    into.rows = body.size() / rowBytes;
    into.values.resize(into.rows * manifest.input.size);
    for (std::size_t i = 0; i < into.values.size(); ++i) {
        into.values[i] = ReadLittleEndianDouble(body.data() + i * sizeof(double));
    }
    return true;
}

bool DecodeJson(
    std::string_view body,
    const Serving::ModelManifest& manifest,
    const HandlerLimits& limits,
    Decoded& into,
    HttpReply& failure
) {
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(body);
    } catch (const nlohmann::json::exception& error) {
        failure = MalformedBody(Truncated(error.what()));
        return false;
    }

    if (!document.is_object() || !document.contains("inputs")) {
        failure = MalformedBody("body must be an object with an \"inputs\" key");
        return false;
    }
    const auto& inputs = document.at("inputs");
    if (!inputs.is_array()) {
        failure = MalformedBody("\"inputs\" must be an array of rows");
        return false;
    }

    into.rows = inputs.size();
    // Before the reserve, not after: inputs.size() is the client's number and is
    // not bounded by the body's length. A 5 MiB body of empty rows otherwise
    // reserves a gigabyte on the way to its refusal.
    if (into.rows > limits.maxBatchRows) {
        failure = BatchTooLarge(into.rows, limits.maxBatchRows);
        return false;
    }
    into.values.reserve(into.rows * manifest.input.size);
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        const auto& values = inputs[row];
        if (!values.is_array()) {
            failure = MalformedBody("inputs[" + std::to_string(row) + "] must be an array of numbers");
            return false;
        }
        if (values.size() != manifest.input.size) {
            failure = BadInputSize(
                ModelName(manifest) + " takes " + std::to_string(manifest.input.size)
                + " values per row, inputs[" + std::to_string(row) + "] carries "
                + std::to_string(values.size())
            );
            return false;
        }
        for (const auto& value : values) {
            if (!value.is_number()) {
                failure = MalformedBody(
                    "inputs[" + std::to_string(row) + "] must be an array of numbers"
                );
                return false;
            }
            into.values.push_back(value.get<double>());
        }
    }
    return true;
}

}  // namespace

HttpReply ErrorReply(const int status, std::string_view code, std::string_view message) {
    return Reply(status, {{"error", {
        {"code", std::string(code)},
        {"message", std::string(message)},
    }}});
}

HttpReply ListModels(const Serving::RegistrySnapshot& snapshot) {
    nlohmann::json models = nlohmann::json::array();
    for (const auto& [key, model] : snapshot.models) {
        models.push_back(ManifestJson(*model, IsDefault(snapshot, key)));
    }
    return Reply(200, nlohmann::json{
        {"generation", snapshot.generation},
        {"models", std::move(models)},
    });
}

HttpReply DescribeModel(const Serving::RegistrySnapshot& snapshot, std::string_view name) {
    const auto versions = Serving::VersionsOf(snapshot, name);
    if (versions.empty()) {
        return ErrorReply(404, "unknown_model", "no model named \"" + std::string(name) + "\"");
    }
    const auto at = snapshot.defaults.find(name);
    return Reply(200, nlohmann::json{
        {"name", std::string(name)},
        {"versions", versions},
        {"default", at == snapshot.defaults.end() ? nlohmann::json() : nlohmann::json(at->second)},
    });
}

HttpReply DescribeVersion(
    const Serving::RegistrySnapshot& snapshot,
    std::string_view name,
    std::string_view version
) {
    const auto lookup = Serving::Find(snapshot, name, version);
    if (lookup.status != Serving::LookupStatus::Found) {
        return LookupFailure(lookup, name, version);
    }
    return Reply(200, ManifestJson(*lookup.model, IsDefault(snapshot, Serving::ModelKey{
        lookup.model->manifest().name,
        lookup.model->manifest().version,
    })));
}

HttpReply Predict(
    const Serving::RegistrySnapshot& snapshot,
    const PredictRequest& request,
    const HandlerLimits& limits
) {
    const auto lookup = request.version.empty()
        ? Serving::FindDefault(snapshot, request.name)
        : Serving::Find(snapshot, request.name, request.version);
    if (lookup.status != Serving::LookupStatus::Found) {
        return LookupFailure(lookup, request.name, request.version);
    }

    const auto& manifest = lookup.model->manifest();
    const BodyFormat format = FormatOf(request.contentType);
    if (format == BodyFormat::Unsupported) {
        return ErrorReply(415, "unsupported_media_type",
                          std::string("expected ") + kJson + " or " + kBinary);
    }

    Decoded decoded;
    HttpReply failure;
    const bool ok = format == BodyFormat::Binary
        ? DecodeBinary(request.body, manifest, decoded, failure)
        : DecodeJson(request.body, manifest, limits, decoded, failure);
    if (!ok) {
        return failure;
    }

    if (decoded.rows == 0) {
        return BadInputSize("at least one row is required");
    }
    if (decoded.rows > limits.maxBatchRows) {
        return BatchTooLarge(decoded.rows, limits.maxBatchRows);
    }
    for (const double value : decoded.values) {
        if (!std::isfinite(value)) {
            return Reply(400, {{"error", {
                {"code", "bad_input_value"},
                {"message", "every value must be finite"},
            }}});
        }
    }

    // Both decoders establish values.size() == rows * input.size, each in its own
    // way. The fill loop below trusts it, and it runs before Neural::Predict, so
    // the guard there cannot catch a decoder that broke it. One comparison makes
    // the property local instead of an argument about two distant functions.
    if (decoded.values.size() != decoded.rows * manifest.input.size) {
        return ErrorReply(500, "internal", "internal error");
    }

    Matrix input(decoded.rows, manifest.input.size);
    for (std::size_t row = 0; row < decoded.rows; ++row) {
        for (std::size_t col = 0; col < manifest.input.size; ++col) {
            input(row, col) = decoded.values[row * manifest.input.size + col];
        }
    }

    const Matrix output = Neural::Predict(*lookup.model->network(), input);
    const std::size_t width = output.getCols();

    HttpReply reply;
    reply.modelVersion = manifest.version;
    if (format == BodyFormat::Binary) {
        reply.contentType = kBinary;
        reply.body.resize(decoded.rows * width * sizeof(double));
        for (std::size_t row = 0; row < decoded.rows; ++row) {
            for (std::size_t col = 0; col < width; ++col) {
                WriteLittleEndianDouble(
                    reply.body.data() + (row * width + col) * sizeof(double),
                    output(row, col)
                );
            }
        }
        return reply;
    }

    nlohmann::json outputs = nlohmann::json::array();
    for (std::size_t row = 0; row < decoded.rows; ++row) {
        nlohmann::json values = nlohmann::json::array();
        for (std::size_t col = 0; col < width; ++col) {
            values.push_back(output(row, col));
        }
        outputs.push_back(std::move(values));
    }
    reply.contentType = kJson;
    reply.body = nlohmann::json{
        {"model", {{"name", manifest.name}, {"version", manifest.version}}},
        {"outputs", std::move(outputs)},
    }.dump() + "\n";
    return reply;
}

}  // namespace ServingCli
