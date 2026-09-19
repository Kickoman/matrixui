#include <doctest/doctest.h>

#include "core/serving/error.h"
#include "core/serving/manifest.h"

#include "tests/support/model_dir.h"
#include "tests/support/temp_dir.h"

#include <nlohmann/json.hpp>

#include <string>

namespace {

using Serving::ManifestError;
using Serving::ParseManifest;

// Every case here bends one field of a manifest that would otherwise parse.
struct Fixture {
    Tests::TempDir dir;
    std::filesystem::path modelDirectory = Tests::MakeModelDirectory(dir);

    Serving::ModelManifest parse(const nlohmann::json& manifest) const {
        return ParseManifest(manifest, modelDirectory);
    }

    std::string refusal(const nlohmann::json& manifest) const {
        try {
            parse(manifest);
        } catch (const ManifestError& error) {
            return error.what();
        }
        return "<parsed>";
    }
};

nlohmann::json Valid() {
    return Tests::ManifestFor(784, 10);
}

}  // namespace


TEST_CASE("ParseManifest accepts a manifest with only the required fields") {
    const Fixture fixture;
    const auto manifest = fixture.parse(Valid());

    CHECK(manifest.manifestVersion == 1);
    CHECK(manifest.name == "mnist");
    CHECK(manifest.version == "v3");
    CHECK(manifest.input.size == 784);
    CHECK(manifest.output.size == 10);
    CHECK(manifest.input.dtype == "f64");
    CHECK(manifest.input.layout == Serving::InputLayout::Flat);
    CHECK(manifest.output.kind == Serving::OutputKind::Raw);
    CHECK(manifest.input.shape.empty());
    CHECK(manifest.output.labels.empty());
    CHECK_FALSE(manifest.input.range.has_value());
    CHECK(manifest.weights.sha256.empty());
    CHECK(manifest.weights.path == fixture.modelDirectory / "weights.wgt");
    CHECK(manifest.annotations.is_object());
}

TEST_CASE("ParseManifest carries every optional field through") {
    const Fixture fixture;
    auto document = Valid();
    document["weights"]["sha256"] = std::string(64, 'a');
    document["input"]["layout"] = "hwc";
    document["input"]["shape"] = {28, 28, 1};
    document["input"]["range"] = {{"min", 0.0}, {"max", 1.0}};
    document["output"]["kind"] = "classification";
    document["output"]["labels"] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    document["annotations"] = {{"trainedOn", "mnist-8x8"}, {"epochs", 40}};

    const auto manifest = fixture.parse(document);
    CHECK(manifest.weights.sha256 == std::string(64, 'a'));
    CHECK(manifest.input.layout == Serving::InputLayout::Hwc);
    CHECK(manifest.input.shape == std::vector<std::size_t>{28, 28, 1});
    REQUIRE(manifest.input.range.has_value());
    CHECK(manifest.input.range->min == 0.0);
    CHECK(manifest.input.range->max == 1.0);
    CHECK(manifest.output.kind == Serving::OutputKind::Classification);
    CHECK(manifest.output.labels.size() == 10);
    CHECK(manifest.annotations["epochs"] == 40);
}

TEST_CASE("ParseManifest refuses a document that is not an object") {
    const Fixture fixture;
    CHECK_THROWS_AS(fixture.parse(nlohmann::json::array()), ManifestError);
    CHECK_THROWS_AS(fixture.parse(nlohmann::json("text")), ManifestError);
    CHECK_THROWS_AS(fixture.parse(nlohmann::json(7)), ManifestError);
}

TEST_CASE("ParseManifest refuses a manifest version it does not know") {
    const Fixture fixture;
    auto document = Valid();

    document.erase("manifestVersion");
    CHECK(fixture.refusal(document) == "manifest.json: missing required key \"manifestVersion\" in root");

    document = Valid();
    document["manifestVersion"] = 2;
    CHECK(fixture.refusal(document) == "manifest.json: unsupported manifestVersion 2 (supported: 1)");

    document["manifestVersion"] = 0;
    CHECK_THROWS_AS(fixture.parse(document), ManifestError);
    document["manifestVersion"] = "1";
    CHECK_THROWS_AS(fixture.parse(document), ManifestError);
}

TEST_CASE("ParseManifest refuses a key it does not know") {
    const Fixture fixture;

    auto document = Valid();
    document["labels"] = nlohmann::json::array();
    CHECK(std::string(fixture.refusal(document)).find("unknown key \"labels\" in root") != std::string::npos);

    for (const char* section : {"input", "output", "weights"}) {
        auto bent = Valid();
        bent[section]["lables"] = 1;
        CHECK(std::string(fixture.refusal(bent)).find("unknown key \"lables\" in " + std::string(section))
              != std::string::npos);
    }
}

TEST_CASE("ParseManifest keeps annotations opaque but insists it is an object") {
    const Fixture fixture;
    auto document = Valid();

    document["annotations"] = {{"anything", {{"nested", true}}}};
    CHECK(fixture.parse(document).annotations["anything"]["nested"] == true);

    document["annotations"] = 5;
    CHECK_THROWS_AS(fixture.parse(document), ManifestError);
    document["annotations"] = nlohmann::json::array();
    CHECK_THROWS_AS(fixture.parse(document), ManifestError);
}

TEST_CASE("ParseManifest holds name and version to a shape a URL cannot misread") {
    const Fixture fixture;
    for (const char* field : {"name", "version"}) {
        for (const char* bad : {"", "foo/bar", "..", "../admin", "Foo", "-foo", "a b", "%2e%2e"}) {
            auto document = Valid();
            document[field] = bad;
            CHECK_MESSAGE(fixture.refusal(document) != "<parsed>", field << " accepted \"" << bad << "\"");
        }
        auto tooLong = Valid();
        tooLong[field] = std::string(65, 'a');
        CHECK(fixture.refusal(tooLong) != "<parsed>");

        auto justFits = Valid();
        justFits[field] = std::string(64, 'a');
        CHECK(fixture.refusal(justFits) == "<parsed>");
    }
}

TEST_CASE("ParseManifest keeps the weights path inside the model directory") {
    const Fixture fixture;
    auto document = Valid();

    document["weights"].erase("path");
    CHECK(fixture.refusal(document) == "manifest.json: missing required key \"path\" in weights");

    for (const char* bad : {"/etc/passwd", "../other/weights.wgt", ""}) {
        auto bent = Valid();
        bent["weights"]["path"] = bad;
        CHECK_MESSAGE(fixture.refusal(bent) != "<parsed>", "accepted \"" << bad << "\"");
    }

    auto nested = Valid();
    nested["weights"]["path"] = "sub/weights.wgt";
    CHECK(fixture.parse(nested).weights.path == fixture.modelDirectory / "sub" / "weights.wgt");
}

TEST_CASE("ParseManifest refuses a sha256 that is not 64 lowercase hex digits") {
    const Fixture fixture;
    for (const auto& bad : {std::string(63, 'a'), std::string(65, 'a'), std::string(64, 'A'), std::string(64, 'z')}) {
        auto document = Valid();
        document["weights"]["sha256"] = bad;
        CHECK_MESSAGE(fixture.refusal(document) != "<parsed>", "accepted \"" << bad << "\"");
    }
}

TEST_CASE("ParseManifest refuses an input contract that does not add up") {
    const Fixture fixture;

    auto zero = Valid();
    zero["input"]["size"] = 0;
    CHECK(fixture.refusal(zero) == "manifest.json: input.size must be a positive integer, found 0");

    auto text = Valid();
    text["input"]["size"] = "784";
    CHECK_THROWS_AS(fixture.parse(text), ManifestError);

    auto negative = Valid();
    negative["input"]["size"] = -1;
    CHECK_THROWS_AS(fixture.parse(negative), ManifestError);

    auto shape = Valid();
    shape["input"]["shape"] = {28, 28, 3};
    CHECK(std::string(fixture.refusal(shape)).find("multiplies out to 2352, but input.size is 784")
          != std::string::npos);

    auto fits = Valid();
    fits["input"]["shape"] = {28, 28};
    CHECK(fixture.refusal(fits) == "<parsed>");

    auto empty = Valid();
    empty["input"]["shape"] = nlohmann::json::array();
    CHECK_THROWS_AS(fixture.parse(empty), ManifestError);

    auto dtype = Valid();
    dtype["input"]["dtype"] = "f32";
    CHECK(fixture.refusal(dtype) == "manifest.json: input.dtype \"f32\" is not supported (only f64)");

    auto layout = Valid();
    layout["input"]["layout"] = "nchw";
    CHECK_THROWS_AS(fixture.parse(layout), ManifestError);

    auto range = Valid();
    range["input"]["range"] = {{"min", 1.0}, {"max", 0.0}};
    CHECK(std::string(fixture.refusal(range)).find("above max") != std::string::npos);

    auto text_range = Valid();
    text_range["input"]["range"] = {{"min", "0"}, {"max", 1.0}};
    CHECK_THROWS_AS(fixture.parse(text_range), ManifestError);
}

TEST_CASE("ParseManifest refuses labels that do not count out to the output size") {
    const Fixture fixture;

    auto few = Valid();
    few["output"]["labels"] = {"a", "b", "c"};
    CHECK(fixture.refusal(few) == "manifest.json: output.labels has 3 entries, but output.size is 10");

    auto notArray = Valid();
    notArray["output"]["labels"] = "abc";
    CHECK_THROWS_AS(fixture.parse(notArray), ManifestError);

    auto notStrings = Valid();
    notStrings["output"]["labels"] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK_THROWS_AS(fixture.parse(notStrings), ManifestError);

    auto kind = Valid();
    kind["output"]["kind"] = "prophecy";
    CHECK_THROWS_AS(fixture.parse(kind), ManifestError);
}
