#pragma once

#include <QObject>
#include "digits_recognizer.h"


Q_DECLARE_METATYPE(TestResult);


class DigitsRunner : public QObject
{
    Q_OBJECT
public:
    DigitsRunner(DigitsRecognizer recognizer);

public slots:
    void run();
    void requestStop();

signals:
    void updatedStatistics(const TestResult& result) const;

private:
    void updateStatistic(const TestResult& result) const;

    DigitsRecognizer recognizer;
};
