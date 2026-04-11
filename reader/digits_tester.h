#pragma once

#include <QObject>
#include "neural_network_applier.h"


class DigitsTester : public QObject
{
    Q_OBJECT
public:
    struct Info {
        QString networkName;
        unsigned predictedNumber;
        QVector<double> probabilities;
    };

    explicit DigitsTester(QObject* parent = nullptr);
    ~DigitsTester();

public slots:
    void processUpdates(const QImage& image);
    bool loadNetwork(const QString& networkName);

signals:
    void infoUpdated(const Info& prediction);

private:
    Neural::NeuralNetworkApplier network;
    QString networkName;
};
