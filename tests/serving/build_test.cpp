#include <doctest/doctest.h>

#include "core/lib/file_stream.h"
#include "core/serving/build.h"
#include "core/serving/error.h"
#include "core/serving/registry.h"

#include "tests/support/model_dir.h"
#include "tests/support/temp_dir.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace {

using Serving::Build;
using Serving::FailureKind;
using Serving::Find;
using Serving::LookupStatus;
using Serving::ModelFailure;

Serving::RegistryConfig ConfigFor(const std::filesystem::path& root) {
    Serving::RegistryConfig config;
    config.root = root;
    return config;
}

std::vector<std::string> KeysOf(const Serving::RegistrySnapshot& snapshot) {
    std::vector<std::string> keys;
    for (const auto& [key, model] : snapshot.models) {
        keys.push_back(key.name + "/" + key.version);
    }
    return keys;
}

const ModelFailure* FailureFor(const Serving::RegistrySnapshot& snapshot, const std::string& tail) {
    for (const auto& failure : snapshot.failures) {
        if (failure.directory.string().size() >= tail.size()
            && failure.directory.string().compare(
                   failure.directory.string().size() - tail.size(), tail.size(), tail) == 0) {
            return &failure;
        }
    }
    return nullptr;
}

}  // namespace


TEST_CASE("Build refuses a root that is not one") {
    const Tests::TempDir dir;
    CHECK_THROWS_AS(Build(ConfigFor(dir.file("absent"))), Serving::ManifestError);
    CHECK_THROWS_AS(Build(ConfigFor(dir.write("plain.txt", "x"))), Serving::ManifestError);
}

TEST_CASE("Build loads every model directory under the root") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"a", "mnist", "v1", {64, 16, 3}},
        {"b", "mnist", "v3", {64, 16, 3}},
        {"c", "cifar", "v1", {32, 8, 3}},
    });

    const auto snapshot = Build(ConfigFor(root));
    CHECK(KeysOf(*snapshot) == std::vector<std::string>{"cifar/v1", "mnist/v1", "mnist/v3"});
    CHECK(snapshot->failures.empty());
    CHECK(Find(*snapshot, "cifar", "v1").model->network()->inputSize() == 32);
}

TEST_CASE("Build does not depend on the order the filesystem returns entries") {
    const Tests::TempDir dir;
    // Created deliberately out of order: directory_iterator hands entries back
    // in filesystem order, which here is creation order.
    const auto root = Tests::WriteModelTree(dir, {
        {"zebra", "zebra", "v1", {64, 16, 3}},
        {"alpha", "alpha", "v1", {64, 16, 3}},
        {"mnist", "mnist", "v1", {64, 16, 3}},
    });

    CHECK(KeysOf(*Build(ConfigFor(root)))
          == std::vector<std::string>{"alpha/v1", "mnist/v1", "zebra/v1"});
}

TEST_CASE("Build keeps the good models when one directory is bad") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"good-a", "mnist", "v1", {64, 16, 3}},
        {"broken", "mnist", "v2", {64, 16, 3}},
        {"good-b", "mnist", "v3", {64, 16, 3}},
    });
    // Past the 34-byte header, so the refusal is about the weights and carries
    // the byte counts rather than being a header complaint.
    Tests::Truncate(root / "broken" / "weights.wgt", 40);

    const auto snapshot = Build(ConfigFor(root));
    CHECK(KeysOf(*snapshot) == std::vector<std::string>{"mnist/v1", "mnist/v3"});

    const auto* failure = FailureFor(*snapshot, "broken");
    REQUIRE(failure != nullptr);
    CHECK(failure->kind == FailureKind::Integrity);
    // The loader already put the numbers in the message; they must survive.
    CHECK(failure->reason.find("shorter than its header claims") != std::string::npos);
}

TEST_CASE("Build sorts the reasons it could not use a directory") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {{"ok", "mnist", "v1", {64, 16, 3}}});

    std::filesystem::create_directories(root / "empty");                      // Skipped
    std::filesystem::create_directories(root / "bad-json");
    Io::WriteFile(root / "bad-json" / "manifest.json", [](std::ostream& out) { out << "{nope"; });
    std::filesystem::create_directories(root / "contract");
    Tests::WriteManifest(root / "contract", Tests::ManifestFor(100, 3, "other", "v1"));
    Tests::WriteNetwork(root / "contract" / "weights.wgt", {64, 16, 3});

    const auto snapshot = Build(ConfigFor(root));
    CHECK(KeysOf(*snapshot) == std::vector<std::string>{"mnist/v1"});

    REQUIRE(FailureFor(*snapshot, "empty") != nullptr);
    CHECK(FailureFor(*snapshot, "empty")->kind == FailureKind::Skipped);
    CHECK(FailureFor(*snapshot, "bad-json")->kind == FailureKind::Manifest);
    CHECK(FailureFor(*snapshot, "contract")->kind == FailureKind::Contract);
    CHECK(FailureFor(*snapshot, "contract")->reason.find("input.size 100") != std::string::npos);

    CHECK(std::is_sorted(snapshot->failures.begin(), snapshot->failures.end(),
                         [](const ModelFailure& l, const ModelFailure& r) {
                             return l.directory < r.directory;
                         }));
}

TEST_CASE("Build rejects both sides of a name and version collision") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"a-copy", "mnist", "v3", {64, 16, 3}},
        {"mnist-v3", "mnist", "v3", {64, 16, 3}},
        {"other", "mnist", "v1", {64, 16, 3}},
    });

    const auto snapshot = Build(ConfigFor(root));
    // Neither wins -- picking one would make the answer depend on the order the
    // filesystem happened to return.
    CHECK(KeysOf(*snapshot) == std::vector<std::string>{"mnist/v1"});

    const auto* first = FailureFor(*snapshot, "a-copy");
    const auto* second = FailureFor(*snapshot, "mnist-v3");
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(first->kind == FailureKind::Collision);
    CHECK(second->kind == FailureKind::Collision);
    // Both directories are named, so an operator can find them.
    for (const auto* failure : {first, second}) {
        CHECK(failure->reason.find("a-copy") != std::string::npos);
        CHECK(failure->reason.find("mnist-v3") != std::string::npos);
    }
}

TEST_CASE("Build reports a default that names a version it did not load") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"a", "mnist", "v1", {64, 16, 3}},
        {"b", "mnist", "v3", {64, 16, 3}},
    });

    auto config = ConfigFor(root);
    config.defaults = {{"mnist", "v9"}, {"absent", "v1"}};
    const auto snapshot = Build(config);

    // The default is broken, the model is not.
    CHECK(Find(*snapshot, "mnist", "v1").status == LookupStatus::Found);
    CHECK(Serving::FindDefault(*snapshot, "mnist").status == LookupStatus::NoDefaultVersion);
    CHECK(snapshot->defaults.empty());

    const auto dangling = std::count_if(
        snapshot->failures.begin(), snapshot->failures.end(),
        [](const ModelFailure& failure) {
            return failure.reason.find("default version") != std::string::npos;
        });
    CHECK(dangling == 2);
    CHECK(std::any_of(snapshot->failures.begin(), snapshot->failures.end(),
                      [](const ModelFailure& f) { return f.reason.find("loaded: v1, v3") != std::string::npos; }));
}

TEST_CASE("Build carries a default it could resolve into the snapshot") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"a", "mnist", "v1", {64, 16, 3}},
        {"b", "mnist", "v3", {64, 16, 3}},
    });

    auto config = ConfigFor(root);
    config.defaults = {{"mnist", "v3"}};
    const auto snapshot = Build(config);

    // Copied into the snapshot, so composition and defaults swap as one unit.
    CHECK(snapshot->defaults.at("mnist") == "v3");
    CHECK(Serving::FindDefault(*snapshot, "mnist").model->manifest().version == "v3");
}

TEST_CASE("Build does not follow a symbolic link in the root") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {{"mnist-v3", "mnist", "v3", {64, 16, 3}}});

    std::error_code failed;
    std::filesystem::create_directory_symlink(root / "mnist-v3", root / "current", failed);
    if (failed) {
        WARN_MESSAGE(false, "symlinks unavailable here: " << failed.message());
        return;
    }

    // Following it would register one model twice, under two directory names --
    // and models/current -> models/mnist-v3 is a common deployment habit, so the
    // operator has to be able to see why it did nothing.
    const auto snapshot = Build(ConfigFor(root));
    CHECK(KeysOf(*snapshot) == std::vector<std::string>{"mnist/v3"});
    const auto* skipped = FailureFor(*snapshot, "current");
    REQUIRE(skipped != nullptr);
    CHECK(skipped->kind == FailureKind::Skipped);
    CHECK(skipped->reason.find("not followed") != std::string::npos);
}

TEST_CASE("Build refuses a root with more entries than allowed") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {{"a", "mnist", "v1", {64, 16, 3}}});
    for (int i = 0; i < 8; ++i) {
        std::filesystem::create_directories(root / ("filler-" + std::to_string(i)));
    }

    auto config = ConfigFor(root);
    config.maxRootEntries = 4;
    try {
        Build(config);
        FAIL("expected a throw");
    } catch (const Serving::ManifestError& error) {
        CHECK(std::string(error.what()).find("more than the 4 entries allowed") != std::string::npos);
    }
}

TEST_CASE("Build stops loading once the declared weight budget is spent") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"a", "mnist", "v1", {64, 16, 3}},
        {"b", "mnist", "v2", {64, 16, 3}},
        {"c", "mnist", "v3", {64, 16, 3}},
    });

    // (64*16 + 16 + 16*3 + 3) * 8 = 8856 bytes each.
    auto config = ConfigFor(root);
    config.maxDeclaredWeightBytes = 8856 * 2;
    const auto snapshot = Build(config);

    CHECK(snapshot->models.size() == 2);
    const auto* refused = FailureFor(*snapshot, "c");
    REQUIRE(refused != nullptr);
    CHECK(refused->kind == FailureKind::Budget);
    CHECK(refused->reason.find("more than the 17712 allowed") != std::string::npos);
}

TEST_CASE("rebuild leaves the previous composition serving when it throws") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {{"a", "mnist", "v1", {64, 16, 3}}});

    auto config = ConfigFor(root);
    Serving::ModelRegistry registry{config};
    registry.rebuild();
    const auto before = registry.snapshot();
    CHECK(KeysOf(*before) == std::vector<std::string>{"mnist/v1"});

    // A registry whose root has gone away must not drop what it is serving.
    std::filesystem::remove_all(root);
    CHECK_THROWS_AS(registry.rebuild(), Serving::ManifestError);

    CHECK(registry.snapshot().get() == before.get());
    CHECK(Find(*registry.snapshot(), "mnist", "v1").status == LookupStatus::Found);
}

TEST_CASE("rebuild publishes what it could load and keeps the rest as reasons") {
    const Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"good", "mnist", "v1", {64, 16, 3}},
        {"broken", "mnist", "v2", {64, 16, 3}},
    });
    Tests::Truncate(root / "broken" / "weights.wgt", 40);

    Serving::ModelRegistry registry{ConfigFor(root)};
    const auto published = registry.rebuild();

    CHECK(KeysOf(*published) == std::vector<std::string>{"mnist/v1"});
    CHECK(published->failures.size() == 1);
    CHECK(published->generation == 1);

    // Strictness stays available to the caller, one line before publish():
    const auto candidate = Build(ConfigFor(root));
    CHECK_FALSE(candidate->failures.empty());
}
