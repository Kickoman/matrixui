#include "core/words/train/trainer.h"

#include "core/lib/random.h"
#include "core/lib/stream_format.h"
#include "core/words/error.h"
#include "core/words/train/negativesampler.h"
#include "core/words/report/train_report.h"
#include "core/words/train/subsampler.h"
#include "core/words/data/vocabulary.h"
#include "core/words/train/windowsampler.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace Words {

std::vector<Probe> BuildProbeSet(
    const Corpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    const NegativeSampler& negativeSampler,
    const ModelConfig& modelConfig,
    const std::size_t count,
    const std::uint64_t seed
) {
    XorShift rng(seed);
    std::vector<Probe> probes;
    probes.reserve(count);

    constexpr std::size_t segments = 20;
    const std::size_t perSegment = count / segments + 1;
    const std::size_t stride = std::max<std::size_t>(1, corpus.size() / segments);

    for (std::size_t segment = 0; segment < segments && probes.size() < count; ++segment) {
        const std::size_t from = segment * stride;
        if (from >= corpus.size()) {
            break;
        }
        const std::size_t to = std::min(from + 50'000, corpus.size());
        const std::size_t target = std::min(count, probes.size() + perSegment);

        GeneratePairs(corpus, subsampler, windowSampler, rng, [&](const Pair& pair) {
            if (probes.size() >= target) {
                return;
            }
            Probe probe{pair, {}};
            probe.negatives.reserve(modelConfig.negatives);
            for (std::size_t i = 0; i < modelConfig.negatives; ++i) {
                probe.negatives.push_back(negativeSampler.sampleExcluding(pair.context, rng));
            }
            probes.push_back(std::move(probe));
        }, 1000, from, to);
    }

    return probes;
}

double MeanProbeLoss(const SGNSModel& model, const std::vector<Probe>& probes) {
    if (probes.empty()) {
        return 0.;
    }
    double total = 0.;
    for (const auto& probe : probes) {
        total += model.computeLoss(probe.pair, probe.negatives);
    }
    return total / probes.size();
}

std::size_t EstimateTotalPairs(
    const Vocabulary& vocabulary,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    const std::size_t epochs
) {
    const double perEpoch =
        subsampler.getExpectedCorpusLength(vocabulary) * windowSampler.getPairsPerToken();
    return static_cast<std::size_t>(perEpoch * epochs);
}

Trainer::Trainer() = default;
Trainer::~Trainer() = default;

void Trainer::setVocabulary(std::shared_ptr<const Vocabulary> value) {
    if (running.load()) {
        throw Error("cannot change the vocabulary while training is running");
    }
    vocabulary = std::move(value);
}

void Trainer::setCorpus(std::shared_ptr<const Corpus> value) {
    if (running.load()) {
        throw Error("cannot change the corpus while training is running");
    }
    corpus = std::move(value);
}

void Trainer::setModelConfig(const ModelConfig& config) {
    if (running.load()) {
        throw Error("cannot change the model configuration while training is running");
    }
    modelConfig = config;
}

void Trainer::setSamplingConfig(const SamplingConfig& config) {
    if (running.load()) {
        throw Error("cannot change the sampling configuration while training is running");
    }
    samplingConfig = config;
}

void Trainer::setProgressCallback(std::function<void(const TrainProgress&)> callback) {
    progressCallback = std::move(callback);
}

void Trainer::setVerbose(const bool value) {
    verbose = value;
}

void Trainer::setOutputStream(std::ostream* value) {
    stream = value;
}

bool Trainer::isRunning() const {
    return running.load();
}

void Trainer::requestStop() {
    stopRequested.store(true, std::memory_order_relaxed);
}

const Embeddings& Trainer::getInputEmbeddings() const {
    if (!model) {
        throw Error("no model yet -- call train() first");
    }
    return model->getInput();
}

const Embeddings& Trainer::getWordEmbeddings() const {
    if (!model) {
        throw Error("no model yet -- call train() first");
    }
    return model->getSubwordTable().isEnabled() ? wordEmbeddings : model->getInput();
}

std::ostream& Trainer::log() const {
    if (!verbose) {
        return NullStream();
    }
    return stream != nullptr ? *stream : DefaultLogStream();
}

TrainSummary Trainer::train(const TrainConfig& trainConfig) {
    if (running.exchange(true)) {
        throw Error("training is already in progress");
    }

    struct RunningGuard {
        std::atomic<bool>& flag;
        ~RunningGuard() { flag.store(false); }
    } guard{running};

    if (!vocabulary) {
        throw ConfigError("no vocabulary set");
    }
    if (!corpus) {
        throw ConfigError("no corpus set");
    }

    Validate(WordsConfig{modelConfig, samplingConfig, trainConfig}, vocabulary->getSize());

    stopRequested.store(false, std::memory_order_relaxed);
    workerException = nullptr;

    const std::size_t threadCount = trainConfig.threads > 0
        ? trainConfig.threads
        : std::max<std::size_t>(1, std::thread::hardware_concurrency());

    const Subsampler subsampler(*vocabulary, samplingConfig.sample);
    const WindowSampler windowSampler(samplingConfig.window);
    const NegativeSampler negativeSampler(
        *vocabulary, samplingConfig.negativeTableSize, samplingConfig.negativePower);

    XorShift seedRng(trainConfig.seed);
    model = std::make_unique<SGNSModel>(*vocabulary, modelConfig, seedRng);
    wordEmbeddings = Embeddings();

    const auto probes = BuildProbeSet(
        *corpus, subsampler, windowSampler, negativeSampler,
        modelConfig, trainConfig.probePairs, trainConfig.seed);

    const auto total = EstimateTotalPairs(*vocabulary, subsampler, windowSampler, trainConfig.epochs);

    TrainSummary summary;
    summary.threads = threadCount;
    summary.pairsEstimated = total;
    summary.probeCount = probes.size();
    summary.initialLoss = MeanProbeLoss(*model, probes);

    PrintTrainBanner(log(), modelConfig, samplingConfig, trainConfig,
                     threadCount, total, probes.size(), summary.initialLoss);

    std::atomic<std::size_t> processed{0};
    std::atomic<std::size_t> activeWorkers{threadCount};

    const auto started = std::chrono::steady_clock::now();
    const std::size_t perThread = corpus->size() / threadCount;

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (std::size_t index = 0; index < threadCount; ++index) {
        workers.emplace_back([&, index] {
            try {
                const std::size_t from = index * perThread;
                const std::size_t to = (index + 1 == threadCount) ? corpus->size() : from + perThread;

                WorkerContext context(modelConfig, trainConfig.seed + index * 7919 + 1);

                double learningRate = model->getLearningRateForStep(0, total);
                std::size_t sinceSync = 0;

                const auto keepGoing = [this] {
                    return !stopRequested.load(std::memory_order_relaxed);
                };

                for (std::size_t epoch = 0; epoch < trainConfig.epochs && keepGoing(); ++epoch) {
                    GeneratePairsWhile(*corpus, subsampler, windowSampler, context.rng,
                        [&](const Pair& pair) {
                            model->trainPair(pair, learningRate, negativeSampler, context);

                            if (++sinceSync < trainConfig.syncEvery) {
                                return;
                            }
                            const auto done =
                                processed.fetch_add(sinceSync, std::memory_order_relaxed) + sinceSync;
                            sinceSync = 0;
                            learningRate = model->getLearningRateForStep(done, total);
                        },
                        keepGoing, trainConfig.chunkSize, from, to);
                }

                processed.fetch_add(sinceSync, std::memory_order_relaxed);
            } catch (...) {
                const std::lock_guard<std::mutex> lock(exceptionMutex);
                if (!workerException) {
                    workerException = std::current_exception();
                }
                stopRequested.store(true, std::memory_order_relaxed);
            }
            activeWorkers.fetch_sub(1, std::memory_order_release);
        });
    }

    constexpr auto slice = std::chrono::milliseconds(50);
    const auto reportEvery = std::chrono::milliseconds(trainConfig.reportEveryMs);

    std::size_t lastProcessed = 0;
    auto lastReport = started;

    while (activeWorkers.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::min(slice, reportEvery));

        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport < reportEvery) {
            continue;
        }

        const auto done = processed.load(std::memory_order_relaxed);

        TrainProgress progress;
        const double windowSeconds = std::chrono::duration<double>(now - lastReport).count();
        progress.elapsedSeconds = std::chrono::duration<double>(now - started).count();
        progress.pairsDone = done;
        progress.pairsTotal = total;
        progress.pairsPerSecond = windowSeconds > 0.
            ? static_cast<double>(done - lastProcessed) / windowSeconds
            : 0.;
        progress.progress = total > 0 ? std::min(1., static_cast<double>(done) / total) : 0.;
        progress.etaSeconds = progress.pairsPerSecond > 0. && total > done
            ? static_cast<double>(total - done) / progress.pairsPerSecond
            : 0.;
        progress.learningRate = model->getLearningRateForStep(done, total);
        progress.loss = MeanProbeLoss(*model, probes);

        lastProcessed = done;
        lastReport = now;

        summary.history.push_back(progress);
        if (progressCallback) {
            progressCallback(progress);
        }
        PrintTrainProgress(log(), progress);
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (workerException) {
        std::rethrow_exception(workerException);
    }

    summary.pairsDone = processed.load(std::memory_order_relaxed);
    summary.elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    summary.pairsPerSecond = summary.elapsedSeconds > 0.
        ? static_cast<double>(summary.pairsDone) / summary.elapsedSeconds
        : 0.;
    summary.finalLoss = MeanProbeLoss(*model, probes);
    summary.stopped = stopRequested.load(std::memory_order_relaxed);

    if (model->getSubwordTable().isEnabled()) {
        wordEmbeddings = model->composeWords();
    }

    PrintTrainSummary(log(), summary);
    return summary;
}

}  // namespace Words
