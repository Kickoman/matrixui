#pragma once

#include <QObject>
#include "trainer.h"


Q_DECLARE_METATYPE(Neural::TestResult);


class DigitsRecognizerController : public QObject
{
    Q_OBJECT
public:
    struct Info {
        bool initialized;
        bool running;
        std::string pathToTrainingDataset;
        std::string pathToTestingDataset;
        std::string networkName;
        std::vector<std::size_t> layersConfiguration;
    };

    DigitsRecognizerController(Neural::Trainer* recognizer);
    ~DigitsRecognizerController();

    Info getInfo() const;

public slots:
    void run();
    void requestStop();
    void testOnce();
    void loadNetwork(const QString& networkName, const QVector<unsigned>& layers = {});
    bool setTrainingDataset(const QString& pathToDataset);
    bool setTestingDataset(const QString& pathToDataset);


signals:
    void infoUpdated();
    void updatedStatistics(const Neural::TestResult& result) const;

private:
    void updateStatistic(const Neural::TestResult& result) const;

    QString networkName;
    QString pathToTrainingDataset;
    QString pathToTestingDataset;
    Neural::Trainer* recognizer;
};
