#pragma once

#include "core/serving/snapshot.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace ServingCli {

struct HandlerLimits {
    std::size_t maxBatchRows = 64;
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

// For the statuses httplib raises on its own, so every body carries one envelope.
HttpReply ErrorReply(int status, std::string_view code, std::string_view message);

}  // namespace ServingCli
