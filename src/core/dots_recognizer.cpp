#include "core/dots_recognizer.h"
#include "utils/directory_lister.h"

#include <span>


constexpr size_t IMAGE_W = 10;
constexpr size_t IMAGE_H = 10;
constexpr double MIN_LEARNING_RATE = 0.01;
constexpr double MAX_LEARNING_RATE = 1.0;
constexpr double LEARNING_RATE_STEP = 0.01;
constexpr size_t MIN_EPOCHS = 5;
constexpr size_t MAX_EPOCHS = 1000;
constexpr size_t EPOCHS_MULTIPLIER = 2;
constexpr size_t MAX_DOTS_COUNT = IMAGE_H * IMAGE_W / 2;

const DotsRecognizer::TLayers DEFAULT_LAYERS = {
    IMAGE_H * IMAGE_W,
    1000, 1000, 100, 100, 100,
    MAX_DOTS_COUNT + 1  // +1 to represent 0 to MAX_DOTS_COUNT inclusive
};


std::pair<size_t, size_t> Dataset::getIndexAndDotsCount(const std::filesystem::path& path) {
    const auto filename = path.filename().string();
    const auto index = filename.find_first_of('-');
    const auto dotsCount = filename.find_last_of('.');
    return std::make_pair(std::stoi(filename.substr(0, index)), std::stoi(filename.substr(index + 1, dotsCount - index - 1)));
}

bool Dataset::load(const std::filesystem::path& path) {
    const auto files = DirectoryLister::listFilesWithExtensions(
        path / "training",
        {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
    );
    if (files.empty()) {
        return false;
    }
    for (const auto& file : files) {
        const auto [index, dotsCount] = getIndexAndDotsCount(file);
        entries.push_back({file, index, dotsCount});
    }
    loaded = true;
    return true;
}

std::span<const DatasetEntry> Dataset::getBatch(const size_t batchIndex) const {
    const auto start = batchIndex * BATCH_SIZE;
    const auto end = std::min(start + BATCH_SIZE, entries.size());
    return std::span<const DatasetEntry>(entries.begin() + start, entries.begin() + end);
}

DotsRecognizer::TestResult DotsRecognizer::testNetwork() const {
    TestResult testResult(MAX_DOTS_COUNT + 1);
    for (const auto& entry : dataset.entries) {
        if (entry.dotsCount > MAX_DOTS_COUNT) {
            log() << "WARNING! Dots count is greater than max dots count: " << entry.dotsCount << " for " << entry.path << std::endl;
            continue;
        }
        log() << "\r[test] Testing dots count " << entry.dotsCount << " for " << entry.path << std::flush;
        auto &statistics = testResult.getPositionRef(entry.dotsCount);
        ++statistics.totalTests;
        const auto image = PngUtils::fromImage(
            entry.path,
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);
        if (testImage(image, entry.dotsCount)) {
            ++statistics.passedTests;
        }
    }
    log() << std::endl;
    const auto& total = testResult.getTotal();
    if (total.totalTests > 0) {
        log() << "Total rate: " << 100.0 * total.passedTests / total.totalTests << "%" << std::endl;
    } else {
        log() << "Total rate: N/A (no tests run)" << std::endl;
    }
    return testResult;
}

void DotsRecognizer::learnNetwork() {
    auto batch = getBatch(currentBatchIndex);
    if (batch.empty()) {
        currentBatchIndex = getFirstBadBatchIndex();
        batch = getBatch(currentBatchIndex);
        if (batch.empty()) {
            requestStop();
            return;
        }
    }
    trainBatch(batch);
    currentBatchIndex++;
}

const DotsRecognizer::TLayers& DotsRecognizer::getDefaultLayersConfiguration() const {
    return DEFAULT_LAYERS;
}

void DotsRecognizer::setBatchTrainingMode(BatchTrainingMode mode) {
    trainingMode = mode;
}

BatchTrainingMode DotsRecognizer::getBatchTrainingMode() const {
    return trainingMode;
}

DotsRecognizer::Batch DotsRecognizer::getBatch(const size_t batchIndex) const {
    if (!dataset.loaded) {
        if (!dataset.load(getPathToDataset())) {
            throw std::runtime_error("Failed to load dataset");
        }
    }
    return dataset.getBatch(batchIndex);
}

size_t MatrixHash(const Matrix& matrix) {
    size_t hash = 0;
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            hash = hash * 31 + static_cast<size_t>(matrix(i, j) * 1000000);
        }
    }
    return hash;
}

void DotsRecognizer::trainBatch(const Batch& batch) {
    unsigned int epochs = MIN_EPOCHS;
    double learningRate = MIN_LEARNING_RATE;
    do {
        log() << "Training batch with epochs " << epochs << " and learning rate " << learningRate;
        log() << " [mode: " << (trainingMode == BatchTrainingMode::TrueBatch ? "TrueBatch" : "SampleBySample") << "]" << std::endl;

        if (trainingMode == BatchTrainingMode::TrueBatch) {
            trainBatchTrueBatch(batch, epochs, learningRate);
        } else {
            trainBatchSampleBySample(batch, epochs, learningRate);
        }

        if (isStopRequested()) {
            return;
        }

        // Test how many samples pass after training
        size_t passed = 0;
        for (const auto& entry : batch) {
            const auto image = PngUtils::fromImage(
                entry.path,
                IMAGE_H,
                IMAGE_W,
                getPngCache()
            ).transform(1, IMAGE_H * IMAGE_W);
            if (testImage(image, entry.dotsCount)) {
                ++passed;
            }
        }

        log() << "Batch passed " << passed << " out of " << batch.size() << " samples; (" << 100.0 * passed / batch.size() << "%)" << std::endl;
        if (passed == batch.size()) {
            log() << "Batch passed successfully" << std::endl;
            return;
        }
        epochs = std::min(epochs * EPOCHS_MULTIPLIER, MAX_EPOCHS);
        learningRate = std::min(learningRate + LEARNING_RATE_STEP, MAX_LEARNING_RATE);
    } while (!isStopRequested());
}

void DotsRecognizer::trainBatchSampleBySample(const Batch& batch, unsigned int epochs, double learningRate) {
    for (const auto& entry : batch) {
        if (isStopRequested()) {
            return;
        }
        const auto image = PngUtils::fromImage(
            entry.path,
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);
        const auto hash = MatrixHash(image);
        log() << "Training image 0x" << std::hex << hash << std::dec << " with path " << entry.path << std::endl;

        // Skip if already correct
        if (testImage(image, entry.dotsCount)) {
            log() << "  -> Already correct, skipping" << std::endl;
            continue;
        }

        const auto expectedResult = GenerateExpectedResult(entry.dotsCount, MAX_DOTS_COUNT + 1);
        getNetwork().train(image, expectedResult, epochs, learningRate, log());
    }
}

void DotsRecognizer::trainBatchTrueBatch(const Batch& batch, unsigned int epochs, double learningRate) {
    if (batch.empty()) {
        return;
    }

    // Build input matrix: stack all images as rows
    Matrix batchInputs(batch.size(), IMAGE_H * IMAGE_W);
    Matrix batchTargets(batch.size(), MAX_DOTS_COUNT + 1, 0.0);

    size_t row = 0;
    for (const auto& entry : batch) {
        const auto image = PngUtils::fromImage(
            entry.path,
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);

        // Copy image into row of batch matrix
        for (size_t col = 0; col < IMAGE_H * IMAGE_W; ++col) {
            batchInputs(row, col) = image(0, col);
        }

        // Set the target (one-hot encoding)
        if (entry.dotsCount <= MAX_DOTS_COUNT) {
            batchTargets(row, entry.dotsCount) = 1.0;
        }

        ++row;
    }

    log() << "Training on " << batch.size() << " samples simultaneously" << std::endl;

    // Train on entire batch at once
    getNetwork().train(batchInputs, batchTargets, epochs, learningRate, log());
}

bool DotsRecognizer::testImage(const Matrix& image, const size_t expectedIndex) const {
    const auto prediction = GetPredictionFast(getNetwork().predict(image));
    log() << "Predicted: " << prediction << ", Expected: " << expectedIndex << std::endl;
    return prediction == expectedIndex;
}

bool DotsRecognizer::testBatch(const Batch& batch) const {
    for (const auto& entry : batch) {
        const auto image = PngUtils::fromImage(
            entry.path,
            IMAGE_H,
            IMAGE_W,
            getPngCache()
        ).transform(1, IMAGE_H * IMAGE_W);
        if (!testImage(image, entry.dotsCount)) {
            return false;
        }
    }
    return true;
}

size_t DotsRecognizer::getFirstBadBatchIndex() const {
    size_t batchIndex = 0;
    Batch batch;
    do {
        batch = getBatch(batchIndex);
        if (batch.empty() || !testBatch(batch)) {
            return batchIndex;
        }
        ++batchIndex;
    } while (!isStopRequested());
    return batchIndex;
}
