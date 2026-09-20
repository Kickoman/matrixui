#include <doctest/doctest.h>

#include "core/lib/sha256.h"
#include "core/matrix/matrix.h"
#include "core/nn/neural_network_applier.h"
#include "core/serving/error.h"
#include "core/serving/model_loader.h"

#include "tests/support/model_dir.h"
#include "tests/support/temp_dir.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace {

using Serving::ContractError;
using Serving::IntegrityError;
using Serving::LoadModel;
using Serving::ManifestError;

std::string Digest(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return Hash::ToHex(Hash::Sha256OfStream(in));
}

template <typename Exception>
std::string RefusalOf(const std::filesystem::path& directory) {
    try {
        LoadModel(directory);
    } catch (const Exception& error) {
        return error.what();
    }
    return "<loaded>";
}

}  // namespace


TEST_CASE("LoadModel returns a model whose contract matches the network") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});

    const auto model = LoadModel(modelDirectory);

    CHECK(model.manifest().name == "mnist");
    CHECK(model.manifest().version == "v3");
    CHECK(model.directory() == modelDirectory);
    CHECK(model.integrity() == Serving::IntegrityCheck::NotDeclared);
    REQUIRE(model.network() != nullptr);
    CHECK(model.network()->inputSize() == 64);
    CHECK(model.network()->outputSize() == 3);
}

TEST_CASE("LoadModel hands out one network that copies share") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});

    const auto model = LoadModel(modelDirectory);
    const auto copy = model;
    CHECK(copy.network().get() == model.network().get());

    // The loader itself does not cache; keeping one model around is the
    // registry's job, not this one's.
    const auto again = LoadModel(modelDirectory);
    CHECK(again.network().get() != model.network().get());

    // The weights outlive the model they arrived in.
    std::shared_ptr<const Neural::NeuralNetwork> kept;
    {
        const auto temporary = LoadModel(modelDirectory);
        kept = temporary.network();
    }
    CHECK(kept->inputSize() == 64);
}

TEST_CASE("A model loaded once predicts the same as an applier holding a copy") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});
    const auto model = LoadModel(modelDirectory);

    const Matrix input(1, 64, [](const std::size_t, const std::size_t col) {
        return static_cast<double>(col % 7) / 7.;
    });

    Neural::NeuralNetworkApplier applier(*model.network());
    const Matrix viaApplier = applier.predict(input);
    const Matrix viaShared = Neural::Predict(*model.network(), input);

    REQUIRE(viaShared.getCols() == viaApplier.getCols());
    for (std::size_t col = 0; col < viaApplier.getCols(); ++col) {
        CHECK(viaShared(0, col) == viaApplier(0, col));
    }
}

TEST_CASE("LoadModel refuses a directory that is not a model") {
    const Tests::TempDir dir;

    CHECK_THROWS_AS(LoadModel(dir.file("absent")), ManifestError);
    CHECK_THROWS_AS(LoadModel(dir.write("plain.txt", "x")), ManifestError);

    const auto empty = Tests::MakeModelDirectory(dir, "empty");
    CHECK(RefusalOf<ManifestError>(empty).find("No manifest.json in") != std::string::npos);

    const auto broken = Tests::MakeModelDirectory(dir, "broken");
    Io::WriteFile(broken / "manifest.json", [](std::ostream& out) { out << "{nope"; });
    CHECK_THROWS_AS(LoadModel(broken), ManifestError);
}

TEST_CASE("LoadModel refuses a manifest whose weights file is not there") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::MakeModelDirectory(dir);
    Tests::WriteManifest(modelDirectory, Tests::ManifestFor(64, 3));

    CHECK(RefusalOf<ManifestError>(modelDirectory).find("weights.path does not name a file")
          != std::string::npos);
}

TEST_CASE("LoadModel refuses a weights blob that is damaged") {
    const Tests::TempDir dir;

    SUBCASE("truncated mid-weights") {
        const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});
        const auto weights = modelDirectory / "weights.wgt";
        Tests::Truncate(weights, std::filesystem::file_size(weights) - 8);
        CHECK(RefusalOf<IntegrityError>(modelDirectory).find("shorter than its header claims")
              != std::string::npos);
    }

    SUBCASE("emptied") {
        const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3}, "empty-weights");
        Tests::Truncate(modelDirectory / "weights.wgt", 0);
        CHECK_THROWS_AS(LoadModel(modelDirectory), IntegrityError);
    }

    SUBCASE("not a network at all") {
        const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3}, "garbage");
        Io::WriteFile(modelDirectory / "weights.wgt", [](std::ostream& out) { out << "garbage"; },
                      std::ios::binary);
        CHECK(RefusalOf<IntegrityError>(modelDirectory).find("Unsupported network version")
              != std::string::npos);
    }
}

TEST_CASE("LoadModel checks the digest when the manifest declares one") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});
    const auto actual = Digest(modelDirectory / "weights.wgt");

    SUBCASE("a matching digest is reported as verified") {
        auto manifest = Tests::ManifestFor(64, 3);
        manifest["weights"]["sha256"] = actual;
        Tests::WriteManifest(modelDirectory, manifest);
        CHECK(LoadModel(modelDirectory).integrity() == Serving::IntegrityCheck::Verified);
    }

    SUBCASE("a mismatch names both digests") {
        const std::string wrong(64, '0');
        auto manifest = Tests::ManifestFor(64, 3);
        manifest["weights"]["sha256"] = wrong;
        Tests::WriteManifest(modelDirectory, manifest);

        const auto refusal = RefusalOf<IntegrityError>(modelDirectory);
        CHECK(refusal.find(wrong) != std::string::npos);
        CHECK(refusal.find(actual) != std::string::npos);
    }
}

TEST_CASE("LoadModel refuses a contract the network does not honour") {
    const Tests::TempDir dir;

    SUBCASE("input size") {
        const auto modelDirectory = Tests::MakeModelDirectory(dir, "input");
        Tests::WriteManifest(modelDirectory, Tests::ManifestFor(100, 3));
        Tests::WriteNetwork(modelDirectory / "weights.wgt", {64, 16, 3});

        CHECK(RefusalOf<ContractError>(modelDirectory)
              == "manifest declares input.size 100, but the network takes 64");
    }

    SUBCASE("output size, labels and all") {
        const auto modelDirectory = Tests::MakeModelDirectory(dir, "output");
        auto manifest = Tests::ManifestFor(64, 5);
        manifest["output"]["labels"] = {"a", "b", "c", "d", "e"};
        Tests::WriteManifest(modelDirectory, manifest);
        Tests::WriteNetwork(modelDirectory / "weights.wgt", {64, 16, 3});

        CHECK(RefusalOf<ContractError>(modelDirectory)
              == "manifest declares output.size 5, but the network produces 3");
    }
}

TEST_CASE("LoadModel resolves the weights path relative to the manifest") {
    const Tests::TempDir dir;

    SUBCASE("a nested path is allowed") {
        const auto modelDirectory = Tests::MakeModelDirectory(dir, "nested");
        std::filesystem::create_directories(modelDirectory / "sub");
        auto manifest = Tests::ManifestFor(64, 3);
        manifest["weights"]["path"] = "sub/weights.wgt";
        Tests::WriteManifest(modelDirectory, manifest);
        Tests::WriteNetwork(modelDirectory / "sub" / "weights.wgt", {64, 16, 3});

        CHECK(LoadModel(modelDirectory).network()->inputSize() == 64);
    }

    SUBCASE("a relative directory loads the same model as an absolute one") {
        const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3}, "relative");
        std::error_code failed;
        const auto relative = std::filesystem::relative(
            modelDirectory, std::filesystem::current_path(), failed);
        REQUIRE_FALSE(failed);

        CHECK(LoadModel(relative).manifest().weights.path
              == LoadModel(modelDirectory).manifest().weights.path);
    }

    SUBCASE("a symlink pointing out of the directory is refused") {
        const auto modelDirectory = Tests::MakeModelDirectory(dir, "symlink");
        Tests::WriteManifest(modelDirectory, Tests::ManifestFor(64, 3));
        const auto outside = dir.file("outside.wgt");
        Tests::WriteNetwork(outside, {64, 16, 3});

        std::error_code failed;
        std::filesystem::create_symlink(outside, modelDirectory / "weights.wgt", failed);
        if (failed) {
            WARN_MESSAGE(false, "symlinks unavailable here: " << failed.message());
            return;
        }

        CHECK(RefusalOf<ManifestError>(modelDirectory).find("escapes the model directory")
              != std::string::npos);
    }
}


// --- Regressions for the defects found after step 1 shipped ------------------

// The registry hands models out of a snapshot, so a reference return would let
// a caller bind into a temporary that dies with the full expression.
static_assert(
    std::is_same_v<
        decltype(std::declval<const Serving::LoadedModel&>().network()),
        std::shared_ptr<const Neural::NeuralNetwork>>,
    "LoadedModel::network() must return by value");

TEST_CASE("LoadModel reports an unreadable weights blob as an IntegrityError") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});
    const auto weights = modelDirectory / "weights.wgt";

    // The digest has to be declared: without it the only reader is LoadNetwork,
    // whose call is already wrapped, so the case passes even unfixed.
    auto manifest = Tests::ManifestFor(64, 3);
    manifest["weights"]["sha256"] = Digest(weights);
    Tests::WriteManifest(modelDirectory, manifest);

    std::error_code failed;
    std::filesystem::permissions(weights, std::filesystem::perms::none, failed);
    if (failed || std::ifstream(weights).good()) {
        std::filesystem::permissions(weights, std::filesystem::perms::owner_all, failed);
        WARN_MESSAGE(false, "cannot make a file unreadable here");
        return;
    }

    CHECK_THROWS_AS(LoadModel(modelDirectory), IntegrityError);
    std::filesystem::permissions(weights, std::filesystem::perms::owner_all, failed);
}

TEST_CASE("LoadModel refuses a manifest larger than the ceiling") {
    const Tests::TempDir dir;
    const auto modelDirectory = Tests::WriteModelDirectory(dir, {64, 16, 3});

    // A registry turns one unbounded parse into one per directory.
    auto manifest = Tests::ManifestFor(64, 3);
    manifest["annotations"] = {{"filler", std::string(2u << 20, 'x')}};
    Tests::WriteManifest(modelDirectory, manifest);

    const auto refusal = RefusalOf<ManifestError>(modelDirectory);
    CHECK(refusal.find("more than the 1048576 allowed") != std::string::npos);
}

TEST_CASE("LoadModel lets nothing but a Serving::Error out") {
    const Tests::TempDir dir;

    // Every shape of damage the loader can meet, swept in one place: the header
    // promises this exception set, and a leak of Io::Error or filesystem_error
    // would make it a lie for direct callers.
    std::vector<std::filesystem::path> broken;

    broken.push_back(dir.file("absent"));
    broken.push_back(dir.write("plain.txt", "x"));
    broken.push_back(Tests::MakeModelDirectory(dir, "no-manifest"));

    const auto badJson = Tests::MakeModelDirectory(dir, "bad-json");
    Io::WriteFile(badJson / "manifest.json", [](std::ostream& out) { out << "{nope"; });
    broken.push_back(badJson);

    const auto noWeights = Tests::MakeModelDirectory(dir, "no-weights");
    Tests::WriteManifest(noWeights, Tests::ManifestFor(64, 3));
    broken.push_back(noWeights);

    const auto truncated = Tests::WriteModelDirectory(dir, {64, 16, 3}, "truncated");
    Tests::Truncate(truncated / "weights.wgt", 20);
    broken.push_back(truncated);

    const auto mismatch = Tests::WriteModelDirectory(dir, {64, 16, 3}, "mismatch");
    Tests::WriteManifest(mismatch, Tests::ManifestFor(100, 3));
    broken.push_back(mismatch);

    const auto badDigest = Tests::WriteModelDirectory(dir, {64, 16, 3}, "bad-digest");
    auto withDigest = Tests::ManifestFor(64, 3);
    withDigest["weights"]["sha256"] = std::string(64, '0');
    Tests::WriteManifest(badDigest, withDigest);
    broken.push_back(badDigest);

    const auto loop = Tests::MakeModelDirectory(dir, "symlink-loop");
    Tests::WriteManifest(loop, Tests::ManifestFor(64, 3));
    std::error_code ignored;
    std::filesystem::create_symlink(loop / "other.wgt", loop / "weights.wgt", ignored);
    std::filesystem::create_symlink(loop / "weights.wgt", loop / "other.wgt", ignored);
    if (!ignored) {
        broken.push_back(loop);
    }

    for (const auto& directory : broken) {
        try {
            LoadModel(directory);
            FAIL("expected a throw for " << directory.string());
        } catch (const Serving::Error&) {
            // as documented
        } catch (const std::exception& error) {
            FAIL(directory.string() << " escaped as " << typeid(error).name()
                 << ": " << error.what());
        }
    }
}
