#pragma once

#include "core/words/config.h"
#include "core/words/data/corpus.h"
#include "core/words/train/model.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <vector>

namespace Words {

class Vocabulary;
class Subsampler;
class WindowSampler;
class NegativeSampler;

struct Probe {
    Pair pair;
    std::vector<TWordId> negatives;
};

std::vector<Probe> BuildProbeSet(
    const Corpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    const NegativeSampler& negativeSampler,
    const ModelConfig& modelConfig,
    std::size_t count,
    std::uint64_t seed
);

double MeanProbeLoss(const SGNSModel& model, const std::vector<Probe>& probes);

std::size_t EstimateTotalPairs(
    const Vocabulary& vocabulary,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    std::size_t epochs
);

struct TrainProgress {
    double progress{0.};
    std::size_t pairsDone{0};
    std::size_t pairsTotal{0};
    double learningRate{0.};
    double loss{0.};
    double pairsPerSecond{0.};
    double elapsedSeconds{0.};
    double etaSeconds{0.};
};

struct TrainSummary {
    std::size_t threads{0};
    std::size_t pairsDone{0};
    std::size_t pairsEstimated{0};
    double elapsedSeconds{0.};
    double pairsPerSecond{0.};
    double initialLoss{0.};
    double finalLoss{0.};
    std::size_t probeCount{0};
    bool stopped{false};
    std::vector<TrainProgress> history;
};

class Trainer {
public:
    Trainer();
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    void setVocabulary(std::shared_ptr<const Vocabulary> vocabulary);
    void setCorpus(std::shared_ptr<const Corpus> corpus);
    void setModelConfig(const ModelConfig& config);
    void setSamplingConfig(const SamplingConfig& config);

    void setProgressCallback(std::function<void(const TrainProgress&)> callback);
    void setVerbose(bool verbose);
    void setOutputStream(std::ostream* stream);

    TrainSummary train(const TrainConfig& config = {});

    bool isRunning() const;
    void requestStop();

    const SGNSModel* getModel() const { return model.get(); }
    const Embeddings& getInputEmbeddings() const;
    const Embeddings& getWordEmbeddings() const;

private:
    std::ostream& log() const;

    std::shared_ptr<const Vocabulary> vocabulary;
    std::shared_ptr<const Corpus> corpus;
    ModelConfig modelConfig;
    SamplingConfig samplingConfig;

    std::unique_ptr<SGNSModel> model;
    Embeddings wordEmbeddings;

    std::atomic<bool> stopRequested{false};
    std::atomic<bool> running{false};

    bool verbose{true};
    std::function<void(const TrainProgress&)> progressCallback;
    std::ostream* stream{nullptr};

    std::mutex exceptionMutex;
    std::exception_ptr workerException;
};

}  // namespace Words
