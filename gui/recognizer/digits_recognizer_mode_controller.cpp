#include "gui/recognizer/digits_recognizer_mode_controller.h"
#include "gui/recognizer/digit_input_preprocess.h"

#include "core/lib/file_stream.h"
#include "core/nn/neural_network_loader.h"

#include <QImage>

#include <stdexcept>
#include <string>


DigitsRecognizerModeController::DigitsRecognizerModeController(QObject* parent) : ModeController(parent)
{}

DigitsRecognizerModeController::~DigitsRecognizerModeController()
{}

void DigitsRecognizerModeController::processUpdates(const QImage& image) {
    if (!network.isInitialized())
        return;

    const int h = image.height();
    const int w = image.width();
    if (static_cast<long long>(w) * h != reader::kMnistPixels)
        return;

    Matrix matrix(h, w);

    for (int y = 0; y < h; ++y) {
        const uchar* line = image.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const uchar gray = line[x];
            matrix(y, x) = (255.0 - gray) / 255.0;
        }
    }

    matrix = matrix.transform(1, reader::kMnistPixels);

    // processUpdates runs from a slot on every stroke, so an escaping exception
    // would terminate the application rather than surface anywhere.
    Matrix prediction;
    try {
        prediction = network.predict(matrix);
    } catch (const std::exception&) {
        return;
    }

    const std::size_t n = prediction.getCols();
    QVector<double> qtPredictions(static_cast<int>(n));
    for (std::size_t i = 0; i < n; ++i) {
        qtPredictions[static_cast<int>(i)] = prediction(0, i);
    }

    unsigned predictedNumber = 0;
    double maxProbability = qtPredictions[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (qtPredictions[static_cast<int>(i)] > maxProbability) {
            maxProbability = qtPredictions[static_cast<int>(i)];
            predictedNumber = static_cast<unsigned>(i);
        }
    }

    emit infoUpdated({
        .networkName = networkName,
        .predictedNumber = predictedNumber,
        .probabilities = qtPredictions,
    });
}

bool DigitsRecognizerModeController::loadNetwork(const QString& networkName) {
    auto loaded = Io::TryReadFile(networkName.toStdString(), [](std::istream& in) { return Neural::LoadNetwork(in); }, std::ios::binary);
    if (!loaded.has_value())
        return false;
    if (loaded->inputSize() != static_cast<std::size_t>(reader::kMnistPixels)) {
        throw std::invalid_argument(
            "This mode feeds " + std::to_string(reader::kMnistPixels) +
            " values per digit, but the network takes " + std::to_string(loaded->inputSize())
        );
    }
    network.initializeNetwork(std::move(*loaded));
    this->networkName = networkName;
    return true;
}
