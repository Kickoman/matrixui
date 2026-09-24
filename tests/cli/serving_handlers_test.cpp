#include <doctest/doctest.h>

#include "cli/serving/handlers.h"

#include "core/serving/build.h"
#include "core/serving/registry.h"

#include "tests/support/model_dir.h"
#include "tests/support/temp_dir.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

using ServingCli::HandlerLimits;
using ServingCli::HttpReply;
using ServingCli::PredictRequest;

// Three models over two names, plus one carrying everything the projection is
// supposed to withhold.
Serving::RegistrySnapshot Snapshot(
    const Tests::TempDir& dir,
    std::map<std::string, std::string> defaults = {}
) {
    const auto root = Tests::WriteModelTree(dir, {
        {"mnist-v1", "mnist", "v1", {64, 16, 3}},
        {"mnist-v3", "mnist", "v3", {64, 16, 3}},
        {"wide-v1", "wide", "v1", {18, 16, 8}},
    });

    const auto secret = root / "secret-v1";
    std::filesystem::create_directories(secret);
    auto manifest = Tests::ManifestFor(64, 3, "secret", "v1");
    manifest["annotations"] = nlohmann::json{{"trainedOn", "a-private-corpus"}};
    Tests::WriteManifest(secret, manifest);
    Tests::WriteNetwork(secret / "weights.wgt", {64, 16, 3});

    Serving::RegistryConfig config;
    config.root = root;
    config.defaults = std::move(defaults);
    return Serving::Build(config);
}

std::vector<double> Row(const std::size_t width) {
    std::vector<double> values(width);
    for (std::size_t i = 0; i < width; ++i) {
        values[i] = static_cast<double>((i * 37) % 251) / 251.;
    }
    return values;
}

std::string JsonBody(const std::vector<std::vector<double>>& rows) {
    return nlohmann::json{{"inputs", rows}}.dump();
}

std::string BinaryBody(const std::vector<double>& values) {
    std::string body(values.size() * sizeof(double), '\0');
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::memcpy(body.data() + i * sizeof(double), &values[i], sizeof(double));
    }
    return body;
}

nlohmann::json Parsed(const HttpReply& reply) {
    return nlohmann::json::parse(reply.body);
}

std::string Code(const HttpReply& reply) {
    return Parsed(reply).at("error").at("code").get<std::string>();
}

HttpReply PredictJson(const Serving::RegistrySnapshot& snapshot, std::string_view name,
                      std::string_view version, const std::string& body,
                      const HandlerLimits& limits = {}) {
    return ServingCli::Predict(snapshot, PredictRequest{name, version, "application/json", body}, limits);
}

}  // namespace

TEST_CASE("GET /v1/models lists every model with its contract") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir, {{"mnist", "v3"}});
    const auto reply = ServingCli::ListModels(snapshot);

    CHECK(reply.status == 200);
    CHECK(reply.contentType == "application/json");

    const auto body = Parsed(reply);
    CHECK(body.at("generation") == 0);
    REQUIRE(body.at("models").size() == 4);

    const auto& first = body.at("models")[0];
    CHECK(first.at("name") == "mnist");
    CHECK(first.at("version") == "v1");
    CHECK(first.at("isDefault") == false);
    CHECK(first.at("integrity") == "not declared");
    CHECK(first.at("input").at("size") == 64);
    CHECK(first.at("input").at("dtype") == "f64");
    CHECK(first.at("output").at("size") == 3);

    CHECK(body.at("models")[1].at("version") == "v3");
    CHECK(body.at("models")[1].at("isDefault") == true);
}

TEST_CASE("GET /v1/models withholds what belongs to the operator") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);
    const auto reply = ServingCli::ListModels(snapshot);

    CHECK(reply.body.find("annotations") == std::string::npos);
    CHECK(reply.body.find("a-private-corpus") == std::string::npos);
    CHECK(reply.body.find("weights") == std::string::npos);
    CHECK(reply.body.find("sha256") == std::string::npos);
    CHECK(reply.body.find("directory") == std::string::npos);
    CHECK(reply.body.find(dir.file("models").string()) == std::string::npos);
}

TEST_CASE("GET /v1/models/:name indexes the versions it serves") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir, {{"mnist", "v3"}});

    SUBCASE("with a default") {
        const auto reply = ServingCli::DescribeModel(snapshot, "mnist");
        CHECK(reply.status == 200);
        const auto body = Parsed(reply);
        CHECK(body.at("name") == "mnist");
        CHECK(body.at("versions") == nlohmann::json::array({"v1", "v3"}));
        CHECK(body.at("default") == "v3");
    }

    SUBCASE("without a default") {
        const auto reply = ServingCli::DescribeModel(snapshot, "wide");
        CHECK(reply.status == 200);
        CHECK(Parsed(reply).at("default").is_null());
    }

    SUBCASE("no such name") {
        const auto reply = ServingCli::DescribeModel(snapshot, "absent");
        CHECK(reply.status == 404);
        CHECK(Code(reply) == "unknown_model");
    }
}

TEST_CASE("GET on one version answers with that version or names the others") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);

    SUBCASE("found") {
        const auto reply = ServingCli::DescribeVersion(snapshot, "mnist", "v3");
        CHECK(reply.status == 200);
        CHECK(Parsed(reply).at("version") == "v3");
    }

    SUBCASE("no such version") {
        const auto reply = ServingCli::DescribeVersion(snapshot, "mnist", "v9");
        CHECK(reply.status == 404);
        CHECK(Code(reply) == "unknown_version");
        CHECK(Parsed(reply).at("error").at("availableVersions")
              == nlohmann::json::array({"v1", "v3"}));
    }

    SUBCASE("no such model") {
        const auto reply = ServingCli::DescribeVersion(snapshot, "absent", "v1");
        CHECK(reply.status == 404);
        CHECK(Code(reply) == "unknown_model");
        CHECK_FALSE(Parsed(reply).at("error").contains("availableVersions"));
    }
}

TEST_CASE("Predict answers a JSON row with a JSON row") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);
    const auto reply = PredictJson(snapshot, "mnist", "v3", JsonBody({Row(64)}));

    CHECK(reply.status == 200);
    CHECK(reply.contentType == "application/json");
    CHECK(reply.modelVersion == "v3");

    const auto body = Parsed(reply);
    CHECK(body.at("model").at("name") == "mnist");
    CHECK(body.at("model").at("version") == "v3");
    REQUIRE(body.at("outputs").size() == 1);
    CHECK(body.at("outputs")[0].size() == 3);
}

TEST_CASE("Predict answers a binary body with a binary body") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);
    const auto reply = ServingCli::Predict(
        snapshot,
        PredictRequest{"mnist", "v3", "application/octet-stream", BinaryBody(Row(64))},
        {}
    );

    CHECK(reply.status == 200);
    CHECK(reply.contentType == "application/octet-stream");
    REQUIRE(reply.body.size() == 3 * sizeof(double));

    double first = 0;
    std::memcpy(&first, reply.body.data(), sizeof(first));
    CHECK(std::isfinite(first));
    CHECK(first >= 0.);
    CHECK(first <= 1.);
}

TEST_CASE("Predict on a batch answers each row with the single-row answer") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);
    const auto row = Row(64);

    const auto single = Parsed(PredictJson(snapshot, "mnist", "v3", JsonBody({row})));
    const auto batched = Parsed(PredictJson(snapshot, "mnist", "v3", JsonBody({row, row, row})));

    REQUIRE(batched.at("outputs").size() == 3);
    for (const auto& answer : batched.at("outputs")) {
        REQUIRE(answer.size() == single.at("outputs")[0].size());
        for (std::size_t col = 0; col < answer.size(); ++col) {
            CHECK(answer[col].get<double>()
                  == doctest::Approx(single.at("outputs")[0][col].get<double>()).epsilon(1e-14));
        }
    }
}

TEST_CASE("Predict without a version resolves the registry default") {
    Tests::TempDir dir;

    SUBCASE("a default is configured") {
        const auto snapshot = Snapshot(dir, {{"mnist", "v3"}});
        const auto reply = PredictJson(snapshot, "mnist", "", JsonBody({Row(64)}));
        CHECK(reply.status == 200);
        CHECK(reply.modelVersion == "v3");
    }

    SUBCASE("no default is configured") {
        const auto snapshot = Snapshot(dir);
        const auto reply = PredictJson(snapshot, "mnist", "", JsonBody({Row(64)}));
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "no_default_version");
        CHECK(Parsed(reply).at("error").at("availableVersions")
              == nlohmann::json::array({"v1", "v3"}));
    }
}

TEST_CASE("Predict refuses a representation it does not speak") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);

    SUBCASE("an unknown type") {
        const auto reply = ServingCli::Predict(
            snapshot, PredictRequest{"mnist", "v3", "text/csv", "0,1,2"}, {});
        CHECK(reply.status == 415);
        CHECK(Code(reply) == "unsupported_media_type");
    }

    SUBCASE("no type at all") {
        const auto reply = ServingCli::Predict(
            snapshot, PredictRequest{"mnist", "v3", "", JsonBody({Row(64)})}, {});
        CHECK(reply.status == 415);
    }

    SUBCASE("a charset parameter is still JSON") {
        const auto reply = ServingCli::Predict(
            snapshot,
            PredictRequest{"mnist", "v3", "application/json; charset=utf-8", JsonBody({Row(64)})},
            {});
        CHECK(reply.status == 200);
    }
}

TEST_CASE("Predict refuses a body it cannot read") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);

    SUBCASE("not JSON at all") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", "{nope");
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "malformed_body");
    }

    SUBCASE("no inputs key") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", "{\"rows\":[]}");
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "malformed_body");
    }

    SUBCASE("inputs is not an array") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", "{\"inputs\":5}");
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "malformed_body");
    }

    SUBCASE("a row is not an array") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", "{\"inputs\":[5]}");
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "malformed_body");
    }

    SUBCASE("an element is not a number") {
        auto row = Row(64);
        auto body = nlohmann::json{{"inputs", std::vector<std::vector<double>>{row}}};
        body["inputs"][0][7] = "half";
        const auto reply = PredictJson(snapshot, "mnist", "v3", body.dump());
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "malformed_body");
    }
}

TEST_CASE("Predict refuses a row that is not the model's width") {
    // This is the check the whole stack rests on: below it, one column short is a
    // silently wrong answer and one column long reads past the weights, with
    // Eigen's own assert removed by -DNDEBUG. Both sides must be refused, and
    // refused before a Matrix is built.
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);

    SUBCASE("one column short") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", JsonBody({Row(63)}));
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_size");
    }

    SUBCASE("one column long") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", JsonBody({Row(65)}));
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_size");
    }

    SUBCASE("the message names both numbers") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", JsonBody({Row(63)}));
        const auto message = Parsed(reply).at("error").at("message").get<std::string>();
        CHECK(message.find("64") != std::string::npos);
        CHECK(message.find("63") != std::string::npos);
    }

    SUBCASE("one row of a batch is the wrong width") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", JsonBody({Row(64), Row(63)}));
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_size");
    }

    SUBCASE("a binary body that does not divide into rows") {
        const auto reply = ServingCli::Predict(
            snapshot,
            PredictRequest{"mnist", "v3", "application/octet-stream", BinaryBody(Row(65))},
            {});
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_size");
    }
}

TEST_CASE("Predict refuses a request carrying no rows") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);

    SUBCASE("an empty inputs array") {
        const auto reply = PredictJson(snapshot, "mnist", "v3", "{\"inputs\":[]}");
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_size");
    }

    SUBCASE("an empty binary body") {
        const auto reply = ServingCli::Predict(
            snapshot, PredictRequest{"mnist", "v3", "application/octet-stream", ""}, {});
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_size");
    }
}

TEST_CASE("Predict refuses a batch over the ceiling") {
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);
    const std::vector<std::vector<double>> rows(5, Row(64));

    const auto reply = PredictJson(snapshot, "mnist", "v3", JsonBody(rows), HandlerLimits{4});
    CHECK(reply.status == 413);
    CHECK(Code(reply) == "batch_too_large");
    const auto message = Parsed(reply).at("error").at("message").get<std::string>();
    CHECK(message.find("5 rows") != std::string::npos);
    CHECK(message.find("4 allowed") != std::string::npos);
}

TEST_CASE("Predict refuses a value that is not finite") {
    // JSON cannot carry one. nlohmann throws out_of_range.406 on a literal that
    // overflows a double, and NaN and Infinity are not JSON literals at all, so
    // the finiteness guard is there for the binary path, where the client writes
    // the bit pattern itself. The JSON subcase pins that behaviour: a parser that
    // started yielding inf instead would be caught here rather than in a 200 full
    // of NaN.
    Tests::TempDir dir;
    const auto snapshot = Snapshot(dir);

    SUBCASE("JSON refuses an overflowing literal while parsing") {
        std::string body = "{\"inputs\":[[1e400";
        for (int i = 1; i < 64; ++i) {
            body += ",0.5";
        }
        body += "]]}";
        const auto reply = PredictJson(snapshot, "mnist", "v3", body);
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "malformed_body");
        CHECK(Parsed(reply).at("error").at("message").get<std::string>().find("overflow")
              != std::string::npos);
    }

    SUBCASE("a NaN bit pattern on the binary path") {
        auto values = Row(64);
        values[3] = std::nan("");
        const auto reply = ServingCli::Predict(
            snapshot,
            PredictRequest{"mnist", "v3", "application/octet-stream", BinaryBody(values)},
            {});
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_value");
    }

    SUBCASE("an infinity bit pattern on the binary path") {
        auto values = Row(64);
        values[0] = std::numeric_limits<double>::infinity();
        const auto reply = ServingCli::Predict(
            snapshot,
            PredictRequest{"mnist", "v3", "application/octet-stream", BinaryBody(values)},
            {});
        CHECK(reply.status == 400);
        CHECK(Code(reply) == "bad_input_value");
    }
}

TEST_CASE("ErrorReply carries the same envelope as every handler") {
    const auto reply = ServingCli::ErrorReply(404, "not_found", "no such route");
    CHECK(reply.status == 404);
    CHECK(reply.contentType == "application/json");
    CHECK(Code(reply) == "not_found");
    CHECK(Parsed(reply).at("error").at("message") == "no such route");
}
