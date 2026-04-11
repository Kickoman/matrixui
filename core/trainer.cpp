#include "trainer.h"
#include "dataset.h"
#include "learning_config.h"
#include "neural_network.h"
#include <iostream>
#include <ostream>


namespace {

const std::size_t GetPrediction(const Matrix& embedding) {
    assert(embedding.getRows() == 1);
    assert(embedding.getCols() > 0);
    std::size_t result = 0;
    double maxProbability = embedding(0, 0);
    for (std::size_t i = 1; i < embedding.getCols(); ++i) {
        const auto probability = embedding(0, i);
        //std::cerr << probability << " ";
        if (probability > maxProbability) {
            maxProbability = probability;
            result = i;
        }
    }
    //std::cerr << std::endl;
    return result;
}

Matrix GenerateExpected(const std::size_t outputIndex, const std::size_t totalIndices) {
    auto res = Matrix::zeros(1, totalIndices);
    res(0, outputIndex) = 1;
    return res;
}

}


namespace Neural {

void Trainer::setNetwork(const NeuralNetwork& network) {
    this->network.initializeNetwork(network);
}

void Trainer::setTrainingDataset(std::unique_ptr<Neural::Dataset>&& dataset) {
    trainingDataset = std::move(dataset);
}

void Trainer::setTestingDataset(std::unique_ptr<Neural::Dataset>&& dataset) {
    testingDataset = std::move(dataset);
}

void Trainer::setEpochCallback(const std::function<void()> callback) {
    epochCallback = callback;
}

void Trainer::setVerbose(const bool verbose) {
    this->verbose = verbose;
}

TestResult Trainer::test(std::size_t samplesPerLabelLimit) const {
    const std::size_t outputs = network.getNeuralNetworkConfig().outputSize();

    TestResult result(outputs);
    for (std::size_t outputIdx = 0; outputIdx < outputs && !stopRequested; ++outputIdx) {
        log() << "[test] Testing output " << outputIdx << std::flush;

        TestStatistics statistics;
        const auto samples = testingDataset->getSamplesForLabel(outputIdx, samplesPerLabelLimit);
        for (const auto& sample : samples) {
            if (stopRequested) {
                break;
            }

            const auto prediction = GetPrediction(network.predict(sample.input));
            ++statistics.totalTests;
            statistics.passedTests += (prediction == outputIdx);
        }

        result.setResult(outputIdx, statistics);
        const auto total = result.getTotal();
        if (total.totalTests > 0) {
            log() << " -- current: " << 100.0 * total.passedTests / total.totalTests << "%" << std::endl;
        }
    }
    return result;
}

void Trainer::train(const LearningConfig& config) {
    running.store(true);
    stopRequested.store(false);

    log() << "========== Loading dataset ==========" << std::endl;
    auto allSamples = trainingDataset->getAllSamples(config.datasetLimitPerLabel);
    if (allSamples.empty()) {
        log() << "No training data available." << std::endl;
        running.store(false);
        return;
    }

    NeuralNetwork bestNetwork = network.getNeuralNetworkConfig();
    std::size_t stagnateEpochsCount = 0;
    double bestTrainAccuracy = 0;
    double learningRate = config.initialLearningRate;

    for (std::size_t epoch = 0; epoch < config.maxEpochs && !stopRequested; ++epoch) {
        log() << "========== Epoch " << (epoch + 1) << "/" << config.maxEpochs
              << " (lr=" << learningRate << ") ==========" << std::endl;

        const double trainAccuracy = trainEpoch(allSamples, learningRate, config);
        log() << "Train accuracy: " << trainAccuracy * 100.0 << "%" << std::endl;

        if (trainAccuracy > bestTrainAccuracy + 0.001) {
            bestTrainAccuracy = trainAccuracy;
            bestNetwork = network.getNeuralNetworkConfig();
            stagnateEpochsCount = 0;

            log() << "New best accuracy: " << bestTrainAccuracy * 100.0 << "%" << std::endl;
        } else {
            ++stagnateEpochsCount;
            log() << "No improvement for " << stagnateEpochsCount << "/" << config.patience << " epochs" << std::endl;
        }

        if (epochCallback) {
            epochCallback();
        }

        if (trainAccuracy > 0.99) {
            log() << "Reached 99\%+ accuracy, stopping." << std::endl;
            break;
        }

        if (stagnateEpochsCount >= config.patience) {
            learningRate *= config.learningRateDecay;
            stagnateEpochsCount = 0;
            log() << "Learning rate reduced to " << learningRate << std::endl;
            if (learningRate < config.minLearningRate) {
                log() << "Learning rate is too small (less than " << config.minLearningRate << "), stopping" << std::endl;
                break;
            }

            log() << "Restoring best model." << std::endl;
            network.initializeNetwork(bestNetwork);
        }
    }

    network.initializeNetwork(bestNetwork);
    log() << "Restored best model with accuracy " << bestTrainAccuracy * 100 << "%" << std::endl;
    if (epochCallback) {
        epochCallback();
    }

    running.store(false);
}

bool Trainer::isRunning() const {
    return running.load();
}

void Trainer::requestStop() {
    stopRequested.store(true);
}

const NeuralNetwork& Trainer::getNetwork() const {
    return network.getNeuralNetworkConfig();
}

const Neural::Dataset* Trainer::getTestingDataset() const {
    return testingDataset.get();
}

const Neural::Dataset* Trainer::getTrainingDataset() const {
    return trainingDataset.get();
}

double Trainer::trainEpoch(std::vector<Sample>& samples, const double learningRate, const LearningConfig& config) {
    Neural::Dataset::ShuffleSamples(samples);
    size_t correctCount = 0;
    for (std::size_t i = 0; i < samples.size() && !stopRequested.load(); ++i) {
        const auto& sample = samples[i];
        const auto expected = GenerateExpected(sample.label, network.getNeuralNetworkConfig().outputSize());

        const auto prediction = GetPrediction(network.predict(sample.input));
        for (std::size_t epoch = 0; epoch < config.innerEpochs; ++epoch) {
            network.train(sample.input, expected, learningRate, config.dropoutRate);
        }

        if (prediction == sample.label) {
            ++correctCount;
        }

        if ((i + 1) % 10 == 0 || i + 1 == samples.size()) {
            log() << "\rTraining " << (i + 1) << "/" << samples.size() << " samples. "
                  << "Current accuracy: " << 100 * correctCount / (i + 1) << "%                " << std::flush;
        }
    }

    log() << std::endl;
    return samples.empty() ? 0 : (1. * correctCount / samples.size());
}

std::ostream& Trainer::log() const {
    if (!verbose || !stream) {
        return std::cerr;
    }
    return *stream;
}

void Trainer::setOutputStream(std::ostream* stream) {
    this->stream = stream;
}

}
