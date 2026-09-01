#pragma once

#include "gui/words/words_controller.h"

#include <QWidget>

class QLineEdit;
class QPushButton;
class QSpinBox;

class WordsExploreTabWidget : public QWidget
{
    Q_OBJECT
public:
    explicit WordsExploreTabWidget(QWidget* parent = nullptr);

    void setController(WordsController* controller);
    void updateInfo(const WordsController::Info& info);

private:
    WordsController* controller = nullptr;

    QLineEdit* neighboursWordEdit = nullptr;
    QSpinBox* neighboursCountSpin = nullptr;
    QPushButton* neighboursButton = nullptr;
    QPushButton* batteryButton = nullptr;

    QLineEdit* expressionEdit = nullptr;
    QSpinBox* expressionCountSpin = nullptr;
    QPushButton* expressionButton = nullptr;

    QLineEdit* oddOneEdit = nullptr;
    QPushButton* oddOneButton = nullptr;

    QLineEdit* axisEdit = nullptr;
    QLineEdit* axisWordsEdit = nullptr;
    QSpinBox* axisRestrictSpin = nullptr;
    QSpinBox* axisCountSpin = nullptr;
    QPushButton* axisButton = nullptr;
};
