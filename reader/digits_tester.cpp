#include "digits_tester.h"
#include "neural_network_loader.h"
#include <QImage>
#include <QDebug>
#include <limits>

DigitsTester::DigitsTester(QObject* parent) : QObject(parent)
{}

DigitsTester::~DigitsTester()
{}

void DigitsTester::processUpdates(const QImage& image) {
    qDebug() << "Processing updates";
    int h = image.height();
    int w = image.width();
    qDebug() << "Updates " << h << " " << w;

    Matrix matrix(h, w);

    for (int y = 0; y < h; ++y) {
        const uchar* line = image.scanLine(y);
        for (int x = 0; x < w; ++x) {
            uchar gray = line[x];
            matrix(y, x) = (255.0 - gray) / 255.0;
        }
    }

    matrix = matrix.transform(1, 28 * 28);
    const auto prediction = network.predict(matrix);
    unsigned predictedNumber = 0;
    double maxProbability = prediction(0, 0);
    QVector<double> qtPredictions(prediction.getCols());
    for (std::size_t i = 1; i < prediction.getCols(); ++i) {
        if (prediction(0, i) > maxProbability) {
            maxProbability = prediction(0, i);
            predictedNumber = i;
        }
        qtPredictions[i] = prediction(0, i);
    }
    qDebug() << "Emitting " << predictedNumber;
    QStringList predictions;
    for (std::size_t i = 0; i < prediction.getCols(); ++i) {
        predictions << QString::number(prediction(0, i));
    }
    qDebug() << predictions.join(", ");
    emit infoUpdated({
        .networkName = networkName,
        .predictedNumber = predictedNumber,
        .probabilities = qtPredictions,
    });
}

void DigitsTester::loadNetwork(const QString& networkName) {
    network.initializeNetwork(Neural::LoadNetwork(networkName.toStdString()).value());
}
