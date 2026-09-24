#pragma once

#include "core/serving/snapshot.h"

#include <atomic>
#include <cstddef>
#include <string>
#include <string_view>

namespace Serving {
class ModelRegistry;
}

namespace ServingCli {

struct HandlerLimits {
    std::size_t maxBatchRows = 32;
};

struct PredictRequest {
    std::string_view name;
    std::string_view version;      // empty resolves through the registry default
    std::string_view contentType;
    std::string_view body;
};

struct HttpReply {
    int status = 200;
    std::string contentType;
    std::string body;
    std::string modelVersion;      // X-Model-Version; empty when no model answered
};

HttpReply ListModels(const Serving::RegistrySnapshot& snapshot);
HttpReply DescribeModel(const Serving::RegistrySnapshot& snapshot, std::string_view name);
HttpReply DescribeVersion(
    const Serving::RegistrySnapshot& snapshot,
    std::string_view name,
    std::string_view version
);
HttpReply Predict(
    const Serving::RegistrySnapshot& snapshot,
    const PredictRequest& request,
    const HandlerLimits& limits
);

// Readiness, admin side only. The body has the same shape whether ready or not,
// so a probe never has to branch on the status to read it.
HttpReply Readyz(const Serving::RegistrySnapshot& snapshot, bool strict);

// One rebuild at a time. rebuild() holds the registry's mutex for the whole walk,
// so a second concurrent request would park a connection slot for seconds to pay
// for a re-read the first one is already doing.
struct ReloadGate {
    std::atomic<bool> inFlight{false};
};

HttpReply Reload(Serving::ModelRegistry& registry, ReloadGate& gate);

// For the statuses httplib raises on its own, so every body carries one envelope.
HttpReply ErrorReply(int status, std::string_view code, std::string_view message);

}  // namespace ServingCli
