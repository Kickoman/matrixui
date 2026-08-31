#include <doctest/doctest.h>

#include "core/words/config.h"
#include "core/words/config_json.h"
#include "core/words/error.h"

using namespace Words;

TEST_CASE("Default configuration validates against a normal vocabulary") {
    CHECK_NOTHROW(Validate(WordsConfig{}, 50'000));
}

TEST_CASE("Validate rejects settings that cannot produce a run") {
    WordsConfig config;

    SUBCASE("empty vocabulary") {
        CHECK_THROWS_AS(Validate(config, 0), ConfigError);
    }
    SUBCASE("zero dimension") {
        config.model.dim = 0;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("zero negatives") {
        config.model.negatives = 0;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("non-positive learning rate") {
        config.model.initialLearningRate = 0.;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("zero window") {
        config.sampling.window = 0;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("negative sample threshold") {
        config.sampling.sample = -1.;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("zero epochs") {
        config.train.epochs = 0;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
}

TEST_CASE("Validate catches a negative table smaller than the vocabulary") {
    WordsConfig config;
    config.sampling.negativeTableSize = 1000;

    // This is the failure that otherwise escapes from NegativeSampler's
    // constructor, after a GUI has already started a worker thread.
    CHECK_THROWS_AS(Validate(config, 50'000), ConfigError);
    CHECK_NOTHROW(Validate(config, 500));
}

TEST_CASE("Every ConfigError is also a Words::Error") {
    WordsConfig config;
    config.model.dim = 0;
    CHECK_THROWS_AS(Validate(config, 100), Words::Error);
    CHECK_THROWS_AS(Validate(config, 100), std::runtime_error);
}

TEST_CASE("Configuration survives a JSON round-trip") {
    WordsConfig original;
    original.model.dim = 64;
    original.model.negatives = 7;
    original.sampling.window = 8;
    original.sampling.sample = 5e-4;
    original.train.epochs = 3;
    original.train.seed = 99;

    const nlohmann::json serialized = original;
    const auto restored = serialized.get<WordsConfig>();

    CHECK(restored.model.dim == 64);
    CHECK(restored.model.negatives == 7);
    CHECK(restored.sampling.window == 8);
    CHECK(restored.sampling.sample == doctest::Approx(5e-4));
    CHECK(restored.train.epochs == 3);
    CHECK(restored.train.seed == 99);
}

TEST_CASE("Each config struct serialises independently") {
    const nlohmann::json model = ModelConfig{};
    CHECK(model.contains("dim"));
    CHECK(model.contains("negatives"));

    const nlohmann::json sampling = SamplingConfig{};
    CHECK(sampling.contains("window"));
    CHECK(sampling.contains("sample"));

    const nlohmann::json train = TrainConfig{};
    CHECK(train.contains("epochs"));
    CHECK(train.contains("threads"));
}
