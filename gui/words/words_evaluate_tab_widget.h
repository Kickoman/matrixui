#pragma once

#include "gui/words/words_controller.h"

#include <QWidget>

class QLabel;
class QPushButton;
class QSpinBox;

class WordsEvaluateTabWidget : public QWidget
{
    Q_OBJECT
public:
    explicit WordsEvaluateTabWidget(QWidget* parent = nullptr);

    void setController(WordsController* controller);
    void updateInfo(const WordsController::Info& info);

private:
    WordsController* controller = nullptr;

    QPushButton* chooseAnalogiesButton = nullptr;
    QLabel* analogiesLabel = nullptr;
    QPushButton* chooseSimilarityButton = nullptr;
    QLabel* similarityLabel = nullptr;

    QSpinBox* scoreColumnSpin = nullptr;
    QSpinBox* restrictSpin = nullptr;
    QSpinBox* threadsSpin = nullptr;

    QPushButton* evaluateAnalogiesButton = nullptr;
    QPushButton* evaluateSimilarityButton = nullptr;
};
