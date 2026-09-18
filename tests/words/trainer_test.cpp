#include <doctest/doctest.h>

#include "core/lib/random.h"
#include "core/words/data/corpus.h"
#include "core/words/train/model.h"
#include "core/words/train/negativesampler.h"
#include "core/words/train/subsampler.h"
#include "core/words/train/trainer.h"
#include "core/words/data/vocabulary.h"
#include "core/words/train/windowsampler.h"
#include "tests/support/capture.h"
#include "core/words/error.h"
#include "tests/support/fixtures.h"
#include "tests/support/temp_dir.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>

using namespace Words;

namespace {

constexpr std::size_t kTableSize = 50'000;

struct Pipeline {
    std::string text;
    Vocabulary vocabulary;
    TCorpus corpus;

    explicit Pipeline(const std::size_t distinct = 200, const std::size_t tokens = 40'000)
        : text(Tests::ZipfCorpusText(distinct, tokens))
        , vocabulary(Tests::VocabularyFromText(text, 1))
        , corpus(Tests::CorpusFromText(text, vocabulary))
    {}

    Corpus openCorpus() const { return Corpus(corpus); }

    std::shared_ptr<const Corpus> shareCorpus() const {
        return std::make_shared<const Corpus>(TCorpus(corpus));
    }
};

}  // namespace

TEST_CASE("BuildProbeSet returns the requested number of probes") {
    const Pipeline pipeline;
    const Subsampler subsampler(pipeline.vocabulary, 1e-3);
    const WindowSampler windowSampler(5);
    const NegativeSampler negativeSampler(pipeline.vocabulary, kTableSize);

    ModelConfig config;
    config.dim = 16;
    config.negatives = 5;

    const auto probes = BuildProbeSet(
        pipeline.openCorpus(), subsampler, windowSampler, negativeSampler, config, 300, 42);

    CHECK(probes.size() == 300);
    for (const auto& probe : probes) {
        CHECK(probe.negatives.size() == config.negatives);
        CHECK(probe.pair.center < pipeline.vocabulary.getSize());
        CHECK(probe.pair.context < pipeline.vocabulary.getSize());
    }
}

TEST_CASE("BuildProbeSet is deterministic for a fixed seed") {
    const Pipeline pipeline;
    const Subsampler subsampler(pipeline.vocabulary, 1e-3);
    const WindowSampler windowSampler(5);
    const NegativeSampler negativeSampler(pipeline.vocabulary, kTableSize);

    ModelConfig config;
    config.dim = 16;

    const auto build = [&] {
        return BuildProbeSet(pipeline.openCorpus(), subsampler, windowSampler,
                             negativeSampler, config, 100, 7);
    };

    const auto first = build();
    const auto second = build();

    REQUIRE(first.size() == second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        CHECK(first[i].pair.center == second[i].pair.center);
        CHECK(first[i].pair.context == second[i].pair.context);
        CHECK(first[i].negatives == second[i].negatives);
    }
}

TEST_CASE("MeanProbeLoss on an untouched model is (negatives + 1) * ln 2") {
    const Pipeline pipeline;
    const Subsampler subsampler(pipeline.vocabulary, 1e-3);
    const WindowSampler windowSampler(5);
    const NegativeSampler negativeSampler(pipeline.vocabulary, kTableSize);

    ModelConfig config;
    config.dim = 16;
    config.negatives = 5;

    const auto probes = BuildProbeSet(
        pipeline.openCorpus(), subsampler, windowSampler, negativeSampler, config, 50, 1);

    XorShift rng(3);
    const SGNSModel model(pipeline.vocabulary, config, rng);

    CHECK(MeanProbeLoss(model, probes)
          == doctest::Approx((config.negatives + 1) * std::log(2.)).epsilon(1e-9));
}

TEST_CASE("MeanProbeLoss of an empty probe set is zero") {
    const Pipeline pipeline;
    ModelConfig config;
    config.dim = 8;
    XorShift rng(1);
    const SGNSModel model(pipeline.vocabulary, config, rng);

    CHECK(MeanProbeLoss(model, {}) == doctest::Approx(0.));
}

TEST_CASE("EstimateTotalPairs scales with epochs and the window") {
    const Pipeline pipeline;
    const Subsampler subsampler(pipeline.vocabulary, 1e-3);
    const WindowSampler windowSampler(5);

    const auto one = EstimateTotalPairs(pipeline.vocabulary, subsampler, windowSampler, 1);
    const auto three = EstimateTotalPairs(pipeline.vocabulary, subsampler, windowSampler, 3);

    CHECK(one > 0);
    CHECK(three == doctest::Approx(static_cast<double>(one) * 3).epsilon(0.01));

    // A wider window yields more pairs per token.
    const WindowSampler wider(10);
    CHECK(EstimateTotalPairs(pipeline.vocabulary, subsampler, wider, 1) > one);
}

TEST_CASE("Trainer reduces the mean probe loss") {
    const Pipeline pipeline;

    ModelConfig modelConfig;
    modelConfig.dim = 16;
    modelConfig.negatives = 5;

    SamplingConfig samplingConfig;
    samplingConfig.sample = 1e-3;
    samplingConfig.negativeTableSize = kTableSize;

    TrainConfig trainConfig;
    trainConfig.epochs = 1;
    trainConfig.threads = 1;
    trainConfig.probePairs = 200;
    trainConfig.reportEveryMs = 5;

    Trainer trainer;
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);
    trainer.setSamplingConfig(samplingConfig);
    trainer.setVerbose(false);

    const auto summary = trainer.train(trainConfig);

    CHECK(summary.finalLoss < summary.initialLoss);
    CHECK(summary.pairsDone > 0);
    CHECK_FALSE(summary.stopped);
    CHECK(summary.threads == 1);
    CHECK(summary.probeCount == 200);
}

TEST_CASE("Training is identical on a loaded and a mapped corpus") {
    const Pipeline pipeline;
    const Tests::TempDir dir;

    std::ostringstream stream;
    SaveCorpus(stream, pipeline.corpus);
    const auto path = dir.write("corpus.cor", stream.str());

    const auto trainOnce = [&](const CorpusStorage storage) {
        ModelConfig modelConfig;
        modelConfig.dim = 16;

        SamplingConfig samplingConfig;
        samplingConfig.negativeTableSize = kTableSize;

        TrainConfig trainConfig;
        trainConfig.epochs = 1;
        trainConfig.threads = 1;
        trainConfig.probePairs = 100;

        Trainer trainer;
        trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
        trainer.setCorpus(std::make_shared<const Corpus>(Corpus::Open(path, storage)));
        trainer.setModelConfig(modelConfig);
        trainer.setSamplingConfig(samplingConfig);
        trainer.setVerbose(false);
        return trainer.train(trainConfig);
    };

    const auto loaded = trainOnce(CorpusStorage::Loaded);
    const auto mapped = trainOnce(CorpusStorage::Mapped);

    CHECK(loaded.pairsDone == mapped.pairsDone);
    CHECK(loaded.initialLoss == mapped.initialLoss);
    CHECK(loaded.finalLoss == mapped.finalLoss);
}

TEST_CASE("Trainer routes output to the stream it is given") {
    const Pipeline pipeline(100, 8'000);

    ModelConfig modelConfig;
    modelConfig.dim = 8;

    SamplingConfig samplingConfig;
    samplingConfig.negativeTableSize = kTableSize;

    TrainConfig trainConfig;
    trainConfig.epochs = 1;
    trainConfig.threads = 1;
    trainConfig.probePairs = 50;
    trainConfig.reportEveryMs = 5;

    Trainer trainer;
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);
    trainer.setSamplingConfig(samplingConfig);

    std::ostringstream out;
    trainer.setOutputStream(&out);

    std::string leaked;
    {
        Tests::CoutCapture capture;
        trainer.train(trainConfig);
        leaked = capture.str();
    }

    CHECK(leaked.empty());  // nothing escapes to stdout
    CHECK(out.str().find("dim 8") != std::string::npos);
    CHECK(out.str().find("estimated pairs:") != std::string::npos);
    CHECK(out.str().find("probe set: 50 pairs") != std::string::npos);
    CHECK(out.str().find("final loss:") != std::string::npos);
}

TEST_CASE("setVerbose(false) silences the trainer completely") {
    const Pipeline pipeline(100, 8'000);

    ModelConfig modelConfig;
    modelConfig.dim = 8;
    SamplingConfig samplingConfig;
    samplingConfig.negativeTableSize = kTableSize;
    TrainConfig trainConfig;
    trainConfig.epochs = 1;
    trainConfig.threads = 1;
    trainConfig.probePairs = 20;
    trainConfig.reportEveryMs = 5;

    Trainer trainer;
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);
    trainer.setSamplingConfig(samplingConfig);

    std::ostringstream out;
    trainer.setOutputStream(&out);
    trainer.setVerbose(false);

    trainer.train(trainConfig);

    // Unlike the classifier's trainer, !verbose means silent -- not "print to
    // std::cerr instead".
    CHECK(out.str().empty());
}

TEST_CASE("Trainer reports progress through its callback") {
    const Pipeline pipeline;

    ModelConfig modelConfig;
    modelConfig.dim = 16;
    SamplingConfig samplingConfig;
    samplingConfig.negativeTableSize = kTableSize;
    TrainConfig trainConfig;
    trainConfig.epochs = 2;
    trainConfig.threads = 1;
    trainConfig.probePairs = 50;
    trainConfig.reportEveryMs = 1;

    Trainer trainer;
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);
    trainer.setSamplingConfig(samplingConfig);
    trainer.setVerbose(false);

    std::vector<TrainProgress> ticks;
    trainer.setProgressCallback([&](const TrainProgress& progress) { ticks.push_back(progress); });

    const auto summary = trainer.train(trainConfig);

    CHECK(ticks.size() == summary.history.size());
    for (const auto& tick : ticks) {
        CHECK(tick.progress >= 0.);
        CHECK(tick.progress <= 1.);
        CHECK(tick.pairsTotal == summary.pairsEstimated);
        CHECK(tick.elapsedSeconds >= 0.);
    }
    // Pair counts never go backwards.
    for (std::size_t i = 1; i < ticks.size(); ++i) {
        CHECK(ticks[i].pairsDone >= ticks[i - 1].pairsDone);
    }
}

TEST_CASE("requestStop cuts a run short") {
    const Pipeline pipeline(400, 300'000);

    ModelConfig modelConfig;
    modelConfig.dim = 32;
    SamplingConfig samplingConfig;
    samplingConfig.sample = 0.;  // keep every token, so there is plenty of work
    samplingConfig.negativeTableSize = kTableSize;
    TrainConfig trainConfig;
    trainConfig.epochs = 50;
    trainConfig.threads = 2;
    trainConfig.probePairs = 20;
    trainConfig.reportEveryMs = 1;

    Trainer trainer;
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);
    trainer.setSamplingConfig(samplingConfig);
    trainer.setVerbose(false);

    // Stop as soon as the first progress tick arrives.
    trainer.setProgressCallback([&](const TrainProgress&) { trainer.requestStop(); });

    const auto started = std::chrono::steady_clock::now();
    const auto summary = trainer.train(trainConfig);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    CHECK(summary.stopped);
    // 50 epochs over this corpus would take far longer than a second.
    CHECK(elapsed < 10.);
    CHECK(summary.pairsDone < summary.pairsEstimated);
    CHECK_FALSE(trainer.isRunning());
}

TEST_CASE("Trainer refuses to run without its inputs") {
    Trainer trainer;
    CHECK_THROWS_AS(trainer.train(), ConfigError);

    const Pipeline pipeline(100, 4'000);
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    CHECK_THROWS_AS(trainer.train(), ConfigError);
}

TEST_CASE("Trainer rejects an impossible configuration up front") {
    const Pipeline pipeline(100, 4'000);

    ModelConfig modelConfig;
    modelConfig.dim = 0;  // invalid

    Trainer trainer;
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);
    trainer.setVerbose(false);

    CHECK_THROWS_AS(trainer.train(), ConfigError);
}

TEST_CASE("getInputEmbeddings throws before the first run") {
    Trainer trainer;
    CHECK(trainer.getModel() == nullptr);
    CHECK_THROWS_AS(trainer.getInputEmbeddings(), Words::Error);
}

namespace {

ModelConfig SubwordModelConfig(const std::size_t dim = 16) {
    ModelConfig config;
    config.dim = dim;
    config.negatives = 5;
    config.minN = 3;
    config.maxN = 6;
    config.buckets = 2000;
    return config;
}

void Configure(Trainer& trainer, const Pipeline& pipeline, const ModelConfig& modelConfig) {
    trainer.setVocabulary(std::make_shared<const Vocabulary>(pipeline.vocabulary));
    trainer.setCorpus(pipeline.shareCorpus());
    trainer.setModelConfig(modelConfig);

    SamplingConfig samplingConfig;
    samplingConfig.sample = 1e-3;
    samplingConfig.negativeTableSize = kTableSize;
    trainer.setSamplingConfig(samplingConfig);
    trainer.setVerbose(false);
}

TrainConfig OneEpoch(const std::size_t probes = 100) {
    TrainConfig config;
    config.epochs = 1;
    config.threads = 1;
    config.probePairs = probes;
    config.reportEveryMs = 5;
    return config;
}

}  // namespace

TEST_CASE("Without subwords the word embeddings are the input matrix itself") {
    const Pipeline pipeline;
    ModelConfig modelConfig;
    modelConfig.dim = 16;

    Trainer trainer;
    Configure(trainer, pipeline, modelConfig);
    trainer.train(OneEpoch());

    // No copy and no composition: the same object comes back.
    CHECK(&trainer.getWordEmbeddings() == &trainer.getInputEmbeddings());
}

TEST_CASE("With subwords the word embeddings are the composed matrix") {
    const Pipeline pipeline;
    const auto modelConfig = SubwordModelConfig();

    Trainer trainer;
    Configure(trainer, pipeline, modelConfig);
    trainer.train(OneEpoch());

    const auto& composed = trainer.getWordEmbeddings();
    const auto& raw = trainer.getInputEmbeddings();

    REQUIRE(composed.getWords() == pipeline.vocabulary.getSize());
    REQUIRE(composed.getDim() == modelConfig.dim);
    CHECK(&composed != &raw);

    const auto* model = trainer.getModel();
    REQUIRE(model != nullptr);
    REQUIRE(model->getSubwordTable().isEnabled());

    std::vector<TFloat> hidden;
    bool anyDifference = false;
    for (TWordId id = 0; id < pipeline.vocabulary.getSize(); ++id) {
        CAPTURE(id);
        model->composeInto(id, hidden);
        CHECK(std::equal(hidden.begin(), hidden.end(), composed.row(id)));
        if (!std::equal(hidden.begin(), hidden.end(), raw.row(id))) {
            anyDifference = true;
        }
    }
    // Composition has to actually change something, or the check above would
    // pass on a model that quietly ignored its n-grams.
    CHECK(anyDifference);
}

TEST_CASE("A stopped subword run still materialises its word embeddings") {
    const Pipeline pipeline(400, 300'000);
    auto modelConfig = SubwordModelConfig(32);

    Trainer trainer;
    Configure(trainer, pipeline, modelConfig);
    trainer.setProgressCallback([&](const TrainProgress&) { trainer.requestStop(); });

    TrainConfig trainConfig = OneEpoch(20);
    trainConfig.epochs = 50;
    trainConfig.threads = 2;
    trainConfig.reportEveryMs = 1;

    const auto summary = trainer.train(trainConfig);
    REQUIRE(summary.stopped);

    const auto& composed = trainer.getWordEmbeddings();
    CHECK(composed.getWords() == pipeline.vocabulary.getSize());

    std::vector<TFloat> hidden;
    trainer.getModel()->composeInto(0, hidden);
    CHECK(std::equal(hidden.begin(), hidden.end(), composed.row(0)));
}

TEST_CASE("A second run without subwords does not hand back the previous composition") {
    const Pipeline pipeline;

    Trainer trainer;
    Configure(trainer, pipeline, SubwordModelConfig());
    trainer.train(OneEpoch());
    CHECK(&trainer.getWordEmbeddings() != &trainer.getInputEmbeddings());

    ModelConfig plain;
    plain.dim = 16;
    trainer.setModelConfig(plain);
    trainer.train(OneEpoch());
    CHECK(&trainer.getWordEmbeddings() == &trainer.getInputEmbeddings());
}

TEST_CASE("getWordEmbeddings throws before the first run") {
    Trainer trainer;
    CHECK_THROWS_AS(trainer.getWordEmbeddings(), Words::Error);
}
