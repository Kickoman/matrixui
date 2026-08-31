#pragma once

#include "core/words/config.h"
#include "core/words/corpus.h"
#include "core/words/model.h"

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

// A fixed set of (pair, negatives) samples used to measure loss comparably
// across the run.
struct Probe {
    Pair pair;
    std::vector<TWordId> negatives;
};

std::vector<Probe> BuildProbeSet(
    const TCorpus& corpus,
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

// One progress sample, taken every TrainConfig::reportEveryMs.
struct TrainProgress {
    double progress{0.};             // 0..1
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
    bool stopped{false};                  // requestStop() cut the run short
    std::vector<TrainProgress> history;   // one entry per tick; feeds a chart
};

// Trains SGNS embeddings.
//
// Shaped after Neural::Classifier::Trainer so a GUI controller can drive it the
// same way: configure, attach a progress callback and an output stream, run on
// a worker thread, and cancel with requestStop().
//
// This replaces a free Train() that took seven positional arguments, built its
// own samplers at the call site, wrote to std::cout unconditionally, and could
// not be stopped.
class Trainer {
public:
    Trainer();
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    // --- inputs ---
    // Held by shared_ptr rather than moved in: a GUI keeps showing vocabulary
    // and corpus statistics during and after the run, and the query panel needs
    // the same Vocabulary.
    void setVocabulary(std::shared_ptr<const Vocabulary> vocabulary);
    void setCorpus(std::shared_ptr<const TCorpus> corpus);
    void setModelConfig(const ModelConfig& config);
    void setSamplingConfig(const SamplingConfig& config);

    // --- observation ---
    // The callback runs on the thread that called train(), NOT the caller's UI
    // thread. A Qt controller must marshal it with QMetaObject::invokeMethod.
    void setProgressCallback(std::function<void(const TrainProgress&)> callback);
    void setVerbose(bool verbose);
    void setOutputStream(std::ostream* stream);

    // --- execution ---
    // Blocks until training finishes or requestStop() is honoured. Rethrows any
    // exception raised on a worker thread.
    TrainSummary train(const TrainConfig& config = {});

    bool isRunning() const;
    void requestStop();

    // --- results ---
    const SGNSModel* getModel() const { return model.get(); }
    const Embeddings& getInputEmbeddings() const;

private:
    std::ostream& log() const;

    std::shared_ptr<const Vocabulary> vocabulary;
    std::shared_ptr<const TCorpus> corpus;
    ModelConfig modelConfig;
    SamplingConfig samplingConfig;

    std::unique_ptr<SGNSModel> model;

    std::atomic<bool> stopRequested{false};
    std::atomic<bool> running{false};

    bool verbose{true};
    std::function<void(const TrainProgress&)> progressCallback;
    std::ostream* stream{nullptr};

    std::mutex exceptionMutex;
    std::exception_ptr workerException;
};

}  // namespace Words
