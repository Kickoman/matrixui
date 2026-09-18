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

TEST_CASE("Subword settings are only checked when subwords are enabled") {
    WordsConfig config;

    SUBCASE("disabled by default, so an absurd range is harmless") {
        config.model.minN = 9;
        config.model.maxN = 2;
        CHECK(config.model.buckets == 0);
        CHECK_NOTHROW(Validate(config, 100));
    }
    SUBCASE("an inverted range is rejected once buckets are asked for") {
        config.model.buckets = 1000;
        config.model.minN = 9;
        config.model.maxN = 2;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("a zero min-n is rejected once buckets are asked for") {
        config.model.buckets = 1000;
        config.model.minN = 0;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("more buckets than a 32-bit id can address is rejected") {
        config.model.buckets = std::size_t{1} << 33;
        CHECK_THROWS_AS(Validate(config, 100), ConfigError);
    }
    SUBCASE("a usable subword configuration passes") {
        config.model.buckets = 2'000'000;
        config.model.minN = 3;
        config.model.maxN = 6;
        CHECK_NOTHROW(Validate(config, 100));
    }
}

TEST_CASE("Subwords are off by default") {
    const ModelConfig config;
    CHECK(config.buckets == 0);
    CHECK(config.minN == 3);
    CHECK(config.maxN == 6);
}

TEST_CASE("A configuration stored before subwords existed still parses") {
    // The GUI keeps its config as JSON; a settings blob written by an older
    // build has no subword keys at all and must come back with the defaults
    // rather than throwing.
    const auto stored = nlohmann::json::parse(R"({
        "model": {"dim": 64, "negatives": 7, "initialLearningRate": 0.05, "minLearningRateFactor": 1e-4},
        "sampling": {"window": 8, "sample": 5e-4, "negativeTableSize": 1000000, "negativePower": 0.75},
        "train": {"epochs": 3, "threads": 0, "chunkSize": 1000, "syncEvery": 10000,
                  "reportEveryMs": 3000, "probePairs": 500, "seed": 99}
    })");

    const auto config = stored.get<WordsConfig>();
    CHECK(config.model.dim == 64);
    CHECK(config.model.buckets == 0);
    CHECK(config.model.minN == 3);
    CHECK(config.model.maxN == 6);
}

TEST_CASE("Subword settings survive a JSON round-trip") {
    WordsConfig original;
    original.model.buckets = 500'000;
    original.model.minN = 4;
    original.model.maxN = 5;

    const auto restored = nlohmann::json(original).get<WordsConfig>();
    CHECK(restored.model.buckets == 500'000);
    CHECK(restored.model.minN == 4);
    CHECK(restored.model.maxN == 5);
}
