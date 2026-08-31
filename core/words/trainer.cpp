#include "core/words/trainer.h"

#include "core/lib/random.h"
#include "core/words/negativesampler.h"
#include "core/words/subsampler.h"
#include "core/words/vocabulary.h"
#include "core/words/windowsampler.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

namespace Words {

std::vector<Probe> BuildProbeSet(
    const TCorpus& corpus,
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

void Train(
    SGNSModel& model,
    const TCorpus& corpus,
    const Subsampler& subsampler,
    const WindowSampler& windowSampler,
    const NegativeSampler& negativeSampler,
    const Vocabulary& vocabulary,
    const TrainConfig& trainConfig
) {
    const std::size_t threadCount = trainConfig.threads > 0
        ? trainConfig.threads
        : std::max<std::size_t>(1, std::thread::hardware_concurrency());

    const auto probes = BuildProbeSet(
        corpus, subsampler, windowSampler, negativeSampler,
        model.getConfig(), trainConfig.probePairs, trainConfig.seed);

    const auto total = EstimateTotalPairs(vocabulary, subsampler, windowSampler, trainConfig.epochs);

    std::cout << "dim " << model.getConfig().dim
              << ", negatives " << model.getConfig().negatives
              << ", window " << windowSampler.getWindow()
              << ", epochs " << trainConfig.epochs
              << ", threads " << threadCount << '\n';
    std::cout << "estimated pairs: " << total << '\n';
    std::cout << "probe set: " << probes.size() << " pairs\n";
    std::cout << "initial loss: " << std::fixed << std::setprecision(4)
              << MeanProbeLoss(model, probes) << "\n\n";

    std::atomic<std::size_t> processed{0};
    // std::atomic<bool> finished{false};
    std::atomic<std::size_t> activeWorkers{threadCount};

    const auto started = std::chrono::steady_clock::now();

    const std::size_t perThread = corpus.size() / threadCount;

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (std::size_t index = 0; index < threadCount; ++index) {
        workers.emplace_back([&, index] {
            const std::size_t from = index * perThread;
            const std::size_t to = (index + 1 == threadCount) ? corpus.size() : from + perThread;

            WorkerContext context(model.getConfig(), trainConfig.seed + index * 7919 + 1);

            double learningRate = model.getLearningRateForStep(0, total);
            std::size_t sinceSync = 0;

            for (std::size_t epoch = 0; epoch < trainConfig.epochs; ++epoch) {
                GeneratePairs(corpus, subsampler, windowSampler, context.rng, [&](const Pair& pair) {
                    model.trainPair(pair, learningRate, negativeSampler, context);

                    if (++sinceSync < trainConfig.syncEvery) {
                        return;
                    }
                    const auto done = processed.fetch_add(sinceSync, std::memory_order_relaxed) + sinceSync;
                    sinceSync = 0;
                    learningRate = model.getLearningRateForStep(done, total);
                }, trainConfig.chunkSize, from, to);
            }

            processed.fetch_add(sinceSync, std::memory_order_relaxed);
            activeWorkers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::size_t lastProcessed = 0;
    auto lastReport = started;

    while (activeWorkers.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(trainConfig.reportEveryMs));

        const auto now = std::chrono::steady_clock::now();
        const auto done = processed.load(std::memory_order_relaxed);

        const double windowSeconds = std::chrono::duration<double>(now - lastReport).count();
        const double elapsed = std::chrono::duration<double>(now - started).count();
        const double rate = windowSeconds > 0. ? (done - lastProcessed) / windowSeconds : 0.;
        const double progress = total > 0 ? std::min(1., 1. * done / total) : 0.;
        const double remaining = rate > 0. && total > done ? (total - done) / rate : 0.;

        lastProcessed = done;
        lastReport = now;

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(5) << 100. * progress << "%"
                  << "  pairs " << std::setw(12) << done
                  << "  lr " << std::scientific << std::setprecision(3)
                  << model.getLearningRateForStep(done, total)
                  << "  loss " << std::fixed << std::setprecision(4)
                  << MeanProbeLoss(model, probes)
                  << "  " << std::setprecision(0) << rate / 1000. << "k pairs/s"
                  << "  eta " << remaining << "s"
                  << "  elapsed " << elapsed << "s\n";
    }

    for (auto& worker : workers) {
        worker.join();
    }

    const auto done = processed.load(std::memory_order_relaxed);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    std::cout << "\ndone: " << done << " pairs in " << std::fixed << std::setprecision(1)
              << elapsed << "s (" << done / elapsed / 1000. << "k pairs/s)\n";
    std::cout << "estimate was " << total << ", ratio "
              << std::setprecision(3) << 1. * done / total << '\n';
    std::cout << "final loss: " << std::setprecision(4) << MeanProbeLoss(model, probes) << '\n';
}

}
