#include "core/dots_recognizer.h"
#include "utils/directory_lister.h"

#include <span>


constexpr size_t IMAGE_W = 10;
constexpr size_t IMAGE_H = 10;
constexpr double MIN_LEARNING_RATE = 0.05;
constexpr double MAX_LEARNING_RATE = 1.0;
constexpr double LEARNING_RATE_STEP = 0.05;
constexpr size_t MIN_EPOCHS = 10;
constexpr size_t MAX_EPOCHS = 1000;
constexpr size_t EPOCHS_STEP = 10;
constexpr size_t MAX_DOTS_COUNT = IMAGE_H * IMAGE_W / 2;

const DotsRecognizer::TLayers DEFAULT_LAYERS = {
    IMAGE_H * IMAGE_W,
    1000, 1000, 100,
    MAX_DOTS_COUNT
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
    TestResult testResult(MAX_DOTS_COUNT);
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
    log() << "Total rate: " << 100.0 * testResult.getTotal().passedTests / testResult.getTotal().totalTests << "%" << std::endl;
    return testResult;
}

void DotsRecognizer::learnNetwork() {
    const auto batch = getBatch(currentBatchIndex);
    if (batch.empty()) {
        currentBatchIndex = getFirstBadBatchIndex();
        const auto batch = getBatch(currentBatchIndex);
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

DotsRecognizer::Batch DotsRecognizer::getBatch(const size_t batchIndex) const {
    if (!dataset.loaded) {
        if (!dataset.load(getPathToDataset())) {
            throw std::runtime_error("Failed to load dataset");
        }
    }
    return dataset.getBatch(batchIndex);
}

void DotsRecognizer::trainBatch(const Batch& batch) {
    unsigned int epochs = MIN_EPOCHS;
    double learningRate = MIN_LEARNING_RATE;
    do {
        log() << "Training batch with epochs " << epochs << " and learning rate " << learningRate << std::endl;
        size_t passed = 0;
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
            if (testImage(image, entry.dotsCount)) {
                ++passed;
                continue;
            }
            const auto expectedResult = GenerateExpectedResult(entry.dotsCount, MAX_DOTS_COUNT);
            getNetwork().train(image, expectedResult, epochs, learningRate, log());
        }
        log() << "Batch passed " << passed << " out of " << batch.size() << " samples; (" << 100.0 * passed / batch.size() << "%)" << std::endl;
        if (passed == batch.size()) {
            log() << "Batch passed successfully" << std::endl;
            return;
        }
        epochs = std::min(epochs + EPOCHS_STEP, MAX_EPOCHS);
        learningRate = std::min(learningRate + LEARNING_RATE_STEP, MAX_LEARNING_RATE);
    } while (!isStopRequested());
}

bool DotsRecognizer::testImage(const Matrix& image, const size_t expectedIndex) const {
    const auto prediction = GetPredictionFast(getNetwork().predict(image));
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
