#pragma once

#include "gui/lib/mode_widget.h"

class AdvancedTerminal;
class TimeChart;
class GanConfigWidget;
class DigitsGeneratorController;
class QPushButton;
class QLabel;
class QComboBox;

class DigitsGeneratorModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit DigitsGeneratorModeWidget(QWidget* parent = nullptr);
    ~DigitsGeneratorModeWidget();

    std::ostream* getTerminalStream();

    void setController(DigitsGeneratorController* controller);
    void closeEvent(QCloseEvent* event) override;

private slots:
    void updateInfo();
    void handleEpochCompleted(std::size_t epoch, double dScore, double gScore, double emaReal, double emaGen);
    void handleGenerateClicked();

private:
    std::unique_ptr<std::ostream> terminalStream;
    AdvancedTerminal* terminal = nullptr;
    TimeChart* lossChart = nullptr;

    QPushButton* toggleTrainingButton = nullptr;
    QPushButton* loadClassifierButton = nullptr;
    QPushButton* loadDatasetButton = nullptr;
    QPushButton* loadGeneratorButton = nullptr;
    QPushButton* loadDiscriminatorButton = nullptr;

    QLabel* classifierLabel = nullptr;
    QLabel* datasetLabel = nullptr;
    QLabel* generatorLabel = nullptr;
    QLabel* discriminatorLabel = nullptr;

    GanConfigWidget* ganConfigWidget = nullptr;

    QLabel* previewLabel = nullptr;
    QComboBox* digitSelector = nullptr;
    QPushButton* generateButton = nullptr;

    DigitsGeneratorController* controller = nullptr;
};
