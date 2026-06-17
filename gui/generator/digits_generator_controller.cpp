#include "gui/generator/digits_generator_controller.h"

#include "core/generator/gan_trainer.h"
#include "core/generator/generator.h"
#include "core/generator/discriminator.h"

#include "core/lib/neural_network_applier.h"
#include "core/lib/neural_network_loader.h"
#include "core/lib/neural_network.h"
#include "core/lib/directory_dataset.h"
#include "core/lib/cache.h"
#include "core/lib/matrix_cache.h"

#include "png/pngreader.h"

#include <magic_enum/magic_enum.hpp>

#include <QFile>
#include <QDir>
#include <QDebug>

#include <algorithm>
#include <cmath>
#include <iostream>


namespace {

static const std::vector<std::size_t> DEFAULT_GEN_HIDDEN  = {256, 512};
static const std::vector<std::size_t> DEFAULT_DISC_HIDDEN = {512, 256};

std::vector<Matrix> loadDatasetSamples(
    const QString& path,
    const std::size_t classesCount,
    const std::function<Matrix(const std::filesystem::path& path)>& reader,
    std::size_t limitPerLabel = 0
) {
    Neural::DirectoryDataset dataset(path.toStdString());
    dataset.setFileReader(reader);

    auto samples = dataset.getAllSamples(limitPerLabel);
    Neural::Dataset::FilterSamples(samples, classesCount);
    std::vector<Matrix> images;
    images.reserve(samples.size());
    for (const auto& s : samples)
        images.push_back(s.input);
    return images;
}

Neural::NeuralNetwork makeGeneratorNetwork(
    const std::size_t latentDim,
    const std::size_t numClasses,
    const std::size_t imageHeight,
    const std::size_t imageWidth
) {
    Neural::NeuralNetworkConfiguration cfg;
    cfg.layersSizes = {latentDim + numClasses};
    cfg.layersSizes.insert(cfg.layersSizes.end(), DEFAULT_GEN_HIDDEN.begin(), DEFAULT_GEN_HIDDEN.end());
    cfg.layersSizes.push_back(imageHeight * imageWidth);
    cfg.hiddenActivation = Neural::ActivationType::ReLU;
    cfg.outputActivation = Neural::ActivationType::Sigmoid;
    return Neural::CreateNetwork(cfg);
}

Neural::NeuralNetwork makeDiscriminatorNetwork(
    const std::size_t imageHeight,
    const std::size_t imageWidth
) {
    Neural::NeuralNetworkConfiguration cfg;
    cfg.layersSizes = {imageHeight * imageWidth};
    cfg.layersSizes.insert(cfg.layersSizes.end(), DEFAULT_DISC_HIDDEN.begin(), DEFAULT_DISC_HIDDEN.end());
    cfg.layersSizes.push_back(1);
    cfg.hiddenActivation = Neural::ActivationType::LeakyReLU;
    cfg.outputActivation = Neural::ActivationType::Sigmoid;
    return Neural::CreateNetwork(cfg);
}

QImage matrixToQImage(const Matrix& flat, const std::size_t height, const std::size_t width) {
    QImage img(width, height, QImage::Format_Grayscale8);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            const double val = flat(0, r * width + c);
            const int gray = static_cast<int>(std::clamp(val, 0.0, 1.0) * 255.0);
            img.setPixel(c, r, qRgb(gray, gray, gray));
        }
    }
    return img;
}

} // namespace


DigitsGeneratorController::DigitsGeneratorController(QObject* parent)
    : ModeController(parent)
    , settings("digits_generator")
{
    reader = [this](const std::filesystem::path& path) { return readCached(path); };
}

DigitsGeneratorController::~DigitsGeneratorController() {
    qDebug() << "Digits generator controller destructor";
    waitUntilFinished();
}

void DigitsGeneratorController::loadSettings() {
    generatorPath       = settings.getValue("generator_path",     "generator.wgt").toString();
    discriminatorPath   = settings.getValue("discriminator_path", "discriminator.wgt").toString();
    classifierPath      = settings.getValue("classifier_path",    {}).toString();
    datasetPath         = settings.getValue("dataset_path",       {}).toString();
    imageHeight         = settings.getValue("image_height",       28).toULongLong();
    imageWidth          = settings.getValue("image_width",        28).toULongLong();
    emit infoUpdated();
}

void DigitsGeneratorController::saveSettings() {
    settings.setValue("generator_path",     generatorPath);
    settings.setValue("discriminator_path", discriminatorPath);
    settings.setValue("classifier_path",    classifierPath);
    settings.setValue("dataset_path",       datasetPath);
    settings.setValue("imageWidth",         static_cast<quint64>(imageWidth));
    settings.setValue("imageHeight",        static_cast<quint64>(imageHeight));
}

void DigitsGeneratorController::setLogger(std::ostream* stream) {
    logger = stream;
}

DigitsGeneratorController::Info DigitsGeneratorController::getInfo() const {
    return {
        .running = trainingRunning.load(std::memory_order_relaxed),
        .canRun  = !trainingRunning.load(std::memory_order_relaxed),
        .classifierPath    = classifierPath.toStdString(),
        .datasetPath       = datasetPath.toStdString(),
        .generatorPath     = generatorPath.toStdString(),
        .discriminatorPath = discriminatorPath.toStdString(),
        .imageWidth        = imageWidth,
        .imageHeight       = imageHeight,
        .numClasses        = classifierNet ? classifierNet->outputSize() : 0,
    };
}

QImage DigitsGeneratorController::generateSample(std::size_t label) const {
    if (!generatorNet || !classifierNet) return {};
    const std::size_t numClasses = classifierNet->outputSize();
    const std::size_t inputSize = generatorNet->config.layersSizes.front();
    const std::size_t latentDim = inputSize > numClasses ? inputSize - numClasses : inputSize;
    Neural::GAN::Generator gen(*generatorNet, latentDim, numClasses);
    return matrixToQImage(gen.generate(label), imageHeight, imageWidth);
}

void DigitsGeneratorController::run(const Neural::GAN::GanConfig& config) {
    if (trainingRunning.load(std::memory_order_relaxed)) return;

    if (!loadProject()) {
        return;
    }

    internalRunner = new QThread(this);
    connect(internalRunner, &QThread::finished, internalRunner, &QObject::deleteLater);
    connect(internalRunner, &QThread::finished, this, &DigitsGeneratorController::infoUpdated);

    connect(internalRunner, &QThread::started, [this, config] {
        out() << "Internal runner started..." << std::endl;
        trainingRunning.store(true, std::memory_order_relaxed);
        QMetaObject::invokeMethod(this, &DigitsGeneratorController::infoUpdated);

        const auto numberOfClasses = classifierNet->outputSize();

        const Neural::NeuralNetwork genNet
            = generatorNet.value_or(makeGeneratorNetwork(config.latentDim, numberOfClasses, imageHeight, imageWidth));
        const Neural::NeuralNetwork discNet = discriminatorNet.value_or(makeDiscriminatorNetwork(imageHeight, imageWidth));

        Neural::GAN::Generator     gen (genNet,  config.latentDim, numberOfClasses);
        Neural::GAN::Discriminator disc(discNet);
        Neural::NeuralNetworkApplier cls(*classifierNet);

        Neural::GAN::GanTrainer trainer(std::move(gen), std::move(disc), std::move(cls));
        out() << "Storing active trainer..." << std::endl;
        activeTrainer.store(&trainer, std::memory_order_release);

        trainer.setEpochCallback([this, &trainer](std::size_t epoch, double dScore, double gScore, double emaReal, double emaGen) {
            // Save after each epoch (same pattern as classifier)
            Neural::SaveNetwork(trainer.getGenerator().getNetwork(),     generatorPath.toStdString());
            Neural::SaveNetwork(trainer.getDiscriminator().getNetwork(), discriminatorPath.toStdString());

            QMetaObject::invokeMethod(this, [this, epoch, dScore, gScore, emaReal, emaGen] {
                emit epochCompleted(epoch, dScore, gScore, emaReal, emaGen);
            });
        });

        // Reload with per-run limit if specified, otherwise use pre-loaded samples
        out() << "Loading dataset if necessary..." << std::endl;
        const std::vector<Matrix> samples = config.datasetLimitPerLabel > 0
            ? ::loadDatasetSamples(datasetPath, classifierNet->outputSize(), reader, config.datasetLimitPerLabel)
            : realSamples;

        out() << "Starting training..." << std::endl;
        trainer.train(samples, config, logger);
        out() << "Training finished." << std::endl;

        // Store updated weights back so generateSample works after training
        generatorNet     = trainer.getGenerator().getNetwork();
        discriminatorNet = trainer.getDiscriminator().getNetwork();

        activeTrainer.store(nullptr, std::memory_order_release);
        trainingRunning.store(false, std::memory_order_relaxed);
        out() << "Released mutexes." << std::endl;
        QThread::currentThread()->quit();
    });

    internalRunner->start();
}

void DigitsGeneratorController::requestStop() {
    qDebug() << "Digits generator controller: Requesting stop";
    auto* t = activeTrainer.load(std::memory_order_acquire);
    if (t) t->requestStop();
}

void DigitsGeneratorController::waitUntilFinished() {
    qDebug() << "Digits generator controller: Gracefully waiting";
    if (!internalRunner || !internalRunner->isRunning()) return;
    internalRunner->quit();
    internalRunner->wait();
    qDebug() << "Digits generator controller: Finished";
}

std::ostream& DigitsGeneratorController::out() {
    if (logger) {
        return *logger << "[controller] ";
    }
    return std::cerr << "[controller] ";
}

void DigitsGeneratorController::setClassifierPath(const QString& path) {
    classifierPath = path;
    emit infoUpdated();
}

void DigitsGeneratorController::setGeneratorPath(const QString& path, Neural::NeuralNetworkConfiguration config) {
    generatorPath = path;
    generatorConfiguration = config;
    emit infoUpdated();
}

void DigitsGeneratorController::setDiscriminatorPath(const QString& path, Neural::NeuralNetworkConfiguration config) {
    discriminatorPath = path;
    discriminatorConfiguration = config;
    emit infoUpdated();
}

void DigitsGeneratorController::setDatasetPath(const QString& path) {
    datasetPath = path;
    emit infoUpdated();
}

bool DigitsGeneratorController::loadClassifier() {
    assert(!internalRunner || !internalRunner->isRunning());

    out() << "Loading classifier from " << classifierPath.toStdString() << std::endl;
    auto net = Neural::LoadNetwork(classifierPath.toStdString());
    if (!net) {
        out() << "Couldn't load classifier." << std::endl;
        return false;
    }
    out() << "Classifier loaded:" << std::endl;
    out() << "\tLayers: " << Neural::LayersToTextRepresentation(net->config.layersSizes) << std::endl;
    out() << "\tHidden activation: " << magic_enum::enum_name(net->config.hiddenActivation) << std::endl;
    out() << "\tOutput activation: " << magic_enum::enum_name(net->config.outputActivation) << std::endl;

    classifierNet = std::move(net);
    return true;
}

bool DigitsGeneratorController::loadDataset() {
    assert(!internalRunner || !internalRunner->isRunning());

    if (!classifierNet.has_value()) {
        out() << "Can't load dataset before classifier is loaded!" << std::endl;
        return false;
    }

    out() << "Loading dataset from " << datasetPath.toStdString() << std::endl;
    if (!Neural::DirectoryDataset::IsDirectoryValid(datasetPath.toStdString(), classifierNet->outputSize())) {
        out() << "Couldn't load dataset: directory has invalid format." << std::endl;
        return false;
    }

    realSamples = ::loadDatasetSamples(datasetPath, classifierNet->outputSize(), reader);
    return true;
}

bool DigitsGeneratorController::loadGenerator() {
    assert(!internalRunner || !internalRunner->isRunning());

    if (trainingRunning.load(std::memory_order_relaxed)) {
        out() << "Can't load generator while training is in progress." << std::endl;
        return false;
    }

    if (!classifierNet.has_value()) {
        out() << "Can't load generator before classifier!" << std::endl;
        return false;
    }

    if (!QFile::exists(generatorPath)) {
        out() << "No generator found on path " << generatorPath.toStdString() << std::endl;
        out() << "Creating generator with config:" << std::endl;
        out() << nlohmann::json(generatorConfiguration).dump(2) << std::endl;

        generatorNet = Neural::CreateNetwork(generatorConfiguration);
        return true;
    }
    if (auto network = Neural::LoadNetwork(generatorPath.toStdString())) {
        generatorNet = std::move(network);
        out() << "Successfully loaded generator." << std::endl;
        return true;
    }

    out() << "Couldn't load generator." << std::endl;
    return false;
}

bool DigitsGeneratorController::loadDiscriminator() {
    assert(!internalRunner || !internalRunner->isRunning());

    if (trainingRunning.load(std::memory_order_relaxed)) {
        out() << "Can't load discriminator while training is in progress." << std::endl;
        return false;
    }

    if (!classifierNet.has_value()) {
        out() << "Can't load discriminator before classifier!" << std::endl;
        return false;
    }

    if (!QFile::exists(discriminatorPath)) {
        out() << "No discriminator found on path " << discriminatorPath.toStdString() << std::endl;
        out() << "Creating discriminator with config:" << std::endl;
        out() << nlohmann::json(discriminatorConfiguration).dump(2) << std::endl;

        discriminatorNet = Neural::CreateNetwork(discriminatorConfiguration);
        return true;
    }
    if (auto network = Neural::LoadNetwork(discriminatorPath.toStdString())) {
        discriminatorNet = std::move(network);
        out() << "Successfully loaded discriminator." << std::endl;
        return true;
    }

    out() << "Couldn't load discriminator." << std::endl;
    return false;
}

void DigitsGeneratorController::resetProject() {
    classifierNet.reset();
    realSamples.resize(0);
    discriminatorNet.reset();
    generatorNet.reset();
}

bool DigitsGeneratorController::loadProject() {
    if (trainingRunning.load(std::memory_order_relaxed)) {
        out() << "Can't load project while training is in progress." << std::endl;
        return false;
    }

    if (internalRunner && internalRunner->isRunning()) {
        out() << "Can't load project while runner is running." << std::endl;
        return false;
    }

    // The order is important for validation
    resetProject();
    if (!loadClassifier()) {
        out() << "Classifier is not loaded. Stopping." << std::endl;
        return false;
    }
    if (!loadDataset()) {
        out() << "Dataset is not loaded. Stopping." << std::endl;
        return false;
    }
    if (!loadDiscriminator()) {
        out() << "Discriminator is not loaded. Stopping." << std::endl;
        return false;
    }
    if (!loadGenerator()) {
        out() << "Generator is not loaded. Stopping." << std::endl;
        return false;
    }
    return true;
}

Matrix DigitsGeneratorController::readCached(const std::filesystem::path& path) const {
    static cache::LRUCache<std::tuple<std::filesystem::path, std::size_t, std::size_t>, Matrix, ArbitraryCache::TupleHash> pngCache;
    return ArbitraryCache::DoCached(
        pngCache,
        [](const std::filesystem::path& path, std::size_t height, std::size_t width) -> Matrix {
            return PngUtils::fromImage(path.string(), height, width).transform(1, height * width);
        },
        path, imageHeight, imageWidth
    );
}
