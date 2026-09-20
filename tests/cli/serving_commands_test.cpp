#include <doctest/doctest.h>

#include "cli/serving/commands.h"
#include "cli/serving/options.h"

#include "tests/support/model_dir.h"
#include "tests/support/temp_dir.h"

#include <filesystem>
#include <sstream>
#include <string>

namespace {

ServingCli::ListOptions For(const std::filesystem::path& root) {
    ServingCli::ListOptions options;
    options.root = root.string();
    return options;
}

}  // namespace

TEST_CASE("List prints a table of what loaded") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"b", "mnist", "v3", {64, 16, 10}},
        {"a", "cifar", "v1", {32, 8, 3}},
    });

    std::ostringstream out, err;
    CHECK(ServingCli::List(out, err, For(root)) == ServingCli::kSuccess);
    CHECK(err.str().empty());

    const auto text = out.str();
    CHECK(text.find("MODEL") != std::string::npos);
    CHECK(text.find("cifar") < text.find("mnist"));
    CHECK(text.find("not declared") != std::string::npos);
}

TEST_CASE("List marks the default version") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"a", "mnist", "v1", {64, 16, 10}},
        {"b", "mnist", "v3", {64, 16, 10}},
    });

    auto options = For(root);
    options.defaults = {{"mnist", "v3"}};

    std::ostringstream out, err;
    CHECK(ServingCli::List(out, err, options) == ServingCli::kSuccess);
    CHECK(out.str().find("yes") != std::string::npos);
}

TEST_CASE("List reports a root with nothing in it rather than an empty table") {
    const Tests::TempDir dir;
    const auto root = dir.file("nothing");
    std::filesystem::create_directories(root);

    std::ostringstream out, err;
    CHECK(ServingCli::List(out, err, For(root)) == ServingCli::kSuccess);
    CHECK(out.str().find("No models under") != std::string::npos);
}

TEST_CASE("List separates what loaded from why the rest did not") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"good", "mnist", "v1", {64, 16, 10}},
        {"broken", "mnist", "v2", {64, 16, 10}},
    });
    Tests::Truncate(root / "broken" / "weights.wgt", 40);

    std::ostringstream out, err;
    CHECK(ServingCli::List(out, err, For(root)) == ServingCli::kIncomplete);

    CHECK(out.str().find("mnist") != std::string::npos);
    CHECK(out.str().find("problem") == std::string::npos);

    const auto reasons = err.str();
    CHECK(reasons.find("1 problem:") != std::string::npos);
    CHECK(reasons.find("integrity:") != std::string::npos);
    CHECK(reasons.find("shorter than its header claims") != std::string::npos);
}

TEST_CASE("List indents a reason that runs to several lines") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"one", "dup", "v1", {64, 16, 10}},
        {"two", "dup", "v1", {64, 16, 10}},
    });

    std::ostringstream out, err;
    CHECK(ServingCli::List(out, err, For(root)) == ServingCli::kIncomplete);
    CHECK(err.str().find("\n      ") != std::string::npos);
}

TEST_CASE("List refuses a root it cannot walk") {
    const Tests::TempDir dir;

    std::ostringstream out, err;
    CHECK(ServingCli::List(out, err, For(dir.file("absent"))) == ServingCli::kUnusableRoot);
    CHECK(err.str().find("Not a model root") != std::string::npos);
    CHECK(out.str().empty());
}
