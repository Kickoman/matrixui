#pragma once

#include <QObject>
#include <memory>
#include "digits_recognizer.h"


Q_DECLARE_METATYPE(TestResult);


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


signals:
    void infoUpdated();
    void updatedStatistics(const TestResult& result) const;

private:
    void updateStatistic(const TestResult& result) const;

    DigitsRecognizer* recognizer;
};
