#pragma once

#include <QObject>
#include "core/digits_recognizer.h"


Q_DECLARE_METATYPE(recognition::TestResult);


class DigitsRecognizerController : public QObject
{
    Q_OBJECT
public:
    struct Info {
        bool initialized;
        bool running;
        std::string pathToDataset;
        std::string networkName;
        std::vector<std::size_t> layersConfiguration;
        bool trueBatchMode;
    };

    DigitsRecognizerController(DigitsRecognizer* recognizer);
    ~DigitsRecognizerController();

    Info getInfo() const;

public slots:
    void run();
    void requestStop();
    void loadNetwork(const QString& networkName);
    bool setDataset(const QString& pathToDataset);
    void setSamplesLimit(const unsigned limit);
    void setTestingLimit(const unsigned limit);
    void setBatchTrainingMode(bool trueBatch);


signals:
    void infoUpdated();
    void updatedStatistics(const recognition::TestResult& result) const;

private:
    void updateStatistic(const recognition::TestResult& result) const;

    DigitsRecognizer* recognizer;
};
