#pragma once

#include "gui/lib/mode_controller.h"
#include "core/lib/neural_network_applier.h"


class DigitsRecognizerModeController : public ModeController
{
    Q_OBJECT
public:
    struct Info {
        QString networkName;
        unsigned predictedNumber;
        QVector<double> probabilities;
    };

    explicit DigitsRecognizerModeController(QObject* parent = nullptr);
    ~DigitsRecognizerModeController();

public slots:
    void processUpdates(const QImage& image);
    bool loadNetwork(const QString& networkName);

signals:
    void infoUpdated(const Info& prediction);

private:
    Neural::NeuralNetworkApplier network;
    QString networkName;
};
