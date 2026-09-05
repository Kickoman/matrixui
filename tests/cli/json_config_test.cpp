#include <doctest/doctest.h>

#include "cli/lib/json_config.h"

#include "tests/support/temp_dir.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {

struct TestConfig {
    int alpha = 1;
    double beta = 2.5;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TestConfig, alpha, beta);

}  // namespace

TEST_CASE("LoadJsonConfig treats an empty path as a silent success") {
    std::ostringstream err;
    TestConfig config;
    CHECK(CliLib::LoadJsonConfig(err, "", config, "test"));
    CHECK(err.str().empty());
    CHECK(config.alpha == 1);
    CHECK(config.beta == doctest::Approx(2.5));
}

TEST_CASE("LoadJsonConfig reports a file that cannot be opened") {
    Tests::TempDir dir;
    std::ostringstream err;
    TestConfig config;
    CHECK_FALSE(CliLib::LoadJsonConfig(err, dir.file("absent.json").string(), config, "test"));
    CHECK(err.str().find("Failed to open test config file:") != std::string::npos);
    CHECK(err.str().find("absent.json") != std::string::npos);
    CHECK(config.alpha == 1);
}

TEST_CASE("LoadJsonConfig reports malformed JSON with the label and path") {
    Tests::TempDir dir;
    const auto path = dir.write("bad.json", "{nope");
    std::ostringstream err;
    TestConfig config;
    CHECK_FALSE(CliLib::LoadJsonConfig(err, path.string(), config, "test"));
    CHECK(err.str().find("Failed to parse test config (") != std::string::npos);
    CHECK(err.str().find("bad.json") != std::string::npos);
}

TEST_CASE("LoadJsonConfig replaces the whole config from a complete JSON object") {
    Tests::TempDir dir;
    const auto path = dir.write("good.json", R"({"alpha": 7, "beta": 0.25})");
    std::ostringstream err;
    TestConfig config;
    CHECK(CliLib::LoadJsonConfig(err, path.string(), config, "test"));
    CHECK(err.str().empty());
    CHECK(config.alpha == 7);
    CHECK(config.beta == doctest::Approx(0.25));
}

TEST_CASE("LoadJsonConfig rejects a JSON object missing a field") {
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE reads every field with at(), so a
    // partial config is a parse failure, not a partial override.
    Tests::TempDir dir;
    const auto path = dir.write("partial.json", R"({"alpha": 7})");
    std::ostringstream err;
    TestConfig config;
    CHECK_FALSE(CliLib::LoadJsonConfig(err, path.string(), config, "test"));
    CHECK(err.str().find("Failed to parse test config (") != std::string::npos);
}
