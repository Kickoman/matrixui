#include <doctest/doctest.h>

#include "cli/lib/argv_scan.h"

#include <string>
#include <vector>

namespace {

// Builds a stable argc/argv pair from a list of arguments.
struct Argv {
    explicit Argv(std::vector<std::string> args) : storage(std::move(args)) {
        for (auto& arg : storage) {
            pointers.push_back(arg.data());
        }
    }

    int argc() const { return static_cast<int>(pointers.size()); }
    char** argv() { return pointers.data(); }

    std::vector<std::string> storage;
    std::vector<char*> pointers;
};

}  // namespace

TEST_CASE("FindOptionValue returns the value following a separated flag") {
    Argv args({"prog", "train", "--learning-config", "a.json", "--network", "n.wgt"});
    CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--learning-config") == "a.json");
    CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--network") == "n.wgt");
}

TEST_CASE("FindOptionValue understands the equals form") {
    SUBCASE("plain value") {
        Argv args({"prog", "--learning-config=a.json"});
        CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--learning-config") == "a.json");
    }
    SUBCASE("value containing an equals sign") {
        Argv args({"prog", "--flag=a=b"});
        CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--flag") == "a=b");
    }
    SUBCASE("empty value") {
        Argv args({"prog", "--flag="});
        CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--flag") == "");
    }
}

TEST_CASE("FindOptionValue returns empty when the flag is absent or dangling") {
    SUBCASE("absent") {
        Argv args({"prog", "train", "--other", "x"});
        CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--learning-config") == "");
    }
    SUBCASE("flag is the last argument with no value") {
        Argv args({"prog", "--learning-config"});
        CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--learning-config") == "");
    }
}

TEST_CASE("FindOptionValue never matches argv[0]") {
    Argv args({"--flag", "value"});
    CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--flag") == "");
}

TEST_CASE("FindOptionValue takes the first occurrence when the flag repeats") {
    Argv args({"prog", "--flag", "first", "--flag", "second"});
    CHECK(CliLib::FindOptionValue(args.argc(), args.argv(), "--flag") == "first");
}
