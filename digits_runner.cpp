#include "digits_runner.h"
#include "digits_recognizer.h"


DigitsRunner::DigitsRunner(DigitsRecognizer recognizer)
    : recognizer(recognizer)
{
    this->recognizer.setResultCallback([this](const TestResult& result){
        this->updatedStatistics(result);
    });
}


void DigitsRunner::run() {
    recognizer.doLearning();
}

void DigitsRunner::requestStop() {
    recognizer.requestStop();
}

void DigitsRunner::updateStatistic(const TestResult& result) const {
    emit updatedStatistics(result);
}
