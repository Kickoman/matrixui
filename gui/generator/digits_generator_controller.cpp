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

#include <QFile>
#include <QDir>
#include <QDebug>

#include <algorithm>
#include <cmath>


namespace {

static const std::vector<std::size_t> DEFAULT_GEN_HIDDEN  = {256, 512};
static const std::vector<std::size_t> DEFAULT_DISC_HIDDEN = {512, 256};

bool datasetPathValid(const QString& path) {
    const QDir dir(path);
    for (int i = 0; i < 10; ++i) {
        if (!QDir(dir.filePath(QString::number(i))).exists())
            return false;
    }
    return true;
}

std::vector<Matrix> loadDatasetSamples(
    const QString& path,
    const std::function<Matrix(const std::filesystem::path& path)>& reader,
    std::size_t limitPerLabel = 0
) {
    Neural::DirectoryDataset dataset(path.toStdString());
    dataset.setFileReader(reader);

    const auto samples = dataset.getAllSamples(limitPerLabel);
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
    const QString genPath  = settings.getValue("generator_path",     "generator.wgt").toString();
    const QString discPath = settings.getValue("discriminator_path", "discriminator.wgt").toString();
    const QString clsPath  = settings.getValue("classifier_path",    {}).toString();
    const QString dsPath   = settings.getValue("dataset_path",       {}).toString();
    imageHeight            = settings.getValue("image_height",       28).toULongLong();
    imageWidth             = settings.getValue("image_width",        28).toULongLong();

    loadGenerator(genPath);
    loadDiscriminator(discPath);

    if (!clsPath.isEmpty()) loadClassifier(clsPath);
    if (!dsPath.isEmpty())  loadDataset(dsPath);
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

bool DigitsGeneratorController::canRunTraining() const {
    // Generator and discriminator can be created from scratch, so they're not required.
    return classifierNet.has_value() && !realSamples.empty();
}

DigitsGeneratorController::Info DigitsGeneratorController::getInfo() const {
    return {
        .running = trainingRunning.load(std::memory_order_relaxed),
        .canRun  = canRunTraining(),
        .classifierPath    = classifierPath.toStdString(),
        .datasetPath       = datasetPath.toStdString(),
        .generatorPath     = generatorPath.toStdString(),
        .discriminatorPath = discriminatorPath.toStdString(),
        .imageWidth        = imageWidth,
        .imageHeight       = imageHeight,
    };
}

QImage DigitsGeneratorController::generateSample(std::size_t label) const {
    if (!generatorNet) return {};
    const std::size_t inputSize = generatorNet->config.layersSizes.front();
    const std::size_t numClasses = 10;
    const std::size_t latentDim = inputSize > numClasses ? inputSize - numClasses : inputSize;
    Neural::GAN::Generator gen(*generatorNet, latentDim, numClasses);
    return matrixToQImage(gen.generate(label), imageHeight, imageWidth);
}

void DigitsGeneratorController::run(const Neural::GAN::GanConfig& config) {
    if (!canRunTraining()) return;
    if (trainingRunning.load(std::memory_order_relaxed)) return;

    internalRunner = new QThread(this);
    connect(internalRunner, &QThread::finished, internalRunner, &QObject::deleteLater);

    connect(internalRunner, &QThread::started, [this, config] {
        trainingRunning.store(true, std::memory_order_relaxed);
        QMetaObject::invokeMethod(this, &DigitsGeneratorController::infoUpdated);

        const Neural::NeuralNetwork genNet
            = generatorNet.value_or(makeGeneratorNetwork(config.latentDim, config.numClasses, imageHeight, imageWidth));
        const Neural::NeuralNetwork discNet = discriminatorNet.value_or(makeDiscriminatorNetwork(imageHeight, imageWidth));

        Neural::GAN::Generator     gen (genNet,  config.latentDim, config.numClasses);
        Neural::GAN::Discriminator disc(discNet);
        Neural::NeuralNetworkApplier cls(*classifierNet);

        Neural::GAN::GanTrainer trainer(std::move(gen), std::move(disc), std::move(cls));
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
        const std::vector<Matrix> samples = config.datasetLimitPerLabel > 0
            ? ::loadDatasetSamples(datasetPath, reader, config.datasetLimitPerLabel)
            : realSamples;

        trainer.train(samples, config, logger);

        // Store updated weights back so generateSample works after training
        generatorNet     = trainer.getGenerator().getNetwork();
        discriminatorNet = trainer.getDiscriminator().getNetwork();

        activeTrainer.store(nullptr, std::memory_order_release);
        trainingRunning.store(false, std::memory_order_relaxed);
        QMetaObject::invokeMethod(this, &DigitsGeneratorController::infoUpdated);
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

bool DigitsGeneratorController::loadClassifier(const QString& path) {
    if (internalRunner && internalRunner->isRunning()) return false;
    auto net = Neural::LoadNetwork(path.toStdString());
    if (!net) return false;
    classifierNet = std::move(net);
    classifierPath = path;
    emit infoUpdated();
    return true;
}

bool DigitsGeneratorController::loadDataset(const QString& path) {
    if (internalRunner && internalRunner->isRunning()) return false;
    if (!::datasetPathValid(path)) return false;
    datasetPath = path;
    realSamples = ::loadDatasetSamples(path, reader);
    emit infoUpdated();
    return true;
}

void DigitsGeneratorController::loadGenerator(const QString& path, Neural::NeuralNetworkConfiguration config) {
    if (trainingRunning.load(std::memory_order_relaxed)) return;
    generatorPath = path;
    if (QFile::exists(path)) {
        if (auto net = Neural::LoadNetwork(path.toStdString()))
            generatorNet = std::move(net);
    } else if (!config.layersSizes.empty()) {
        generatorNet = Neural::CreateNetwork(config);
    } else {
        generatorNet = std::nullopt;
    }
    emit infoUpdated();
}

void DigitsGeneratorController::loadDiscriminator(const QString& path, Neural::NeuralNetworkConfiguration config) {
    if (trainingRunning.load(std::memory_order_relaxed)) return;
    discriminatorPath = path;
    if (QFile::exists(path)) {
        if (auto net = Neural::LoadNetwork(path.toStdString()))
            discriminatorNet = std::move(net);
    } else if (!config.layersSizes.empty()) {
        discriminatorNet = Neural::CreateNetwork(config);
    } else {
        discriminatorNet = std::nullopt;
    }
    emit infoUpdated();
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
