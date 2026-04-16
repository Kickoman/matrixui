#pragma once

#include "gui/lib/mode_widget.h"

class AdvancedTerminal;
class AdvancedTerminalStream;
class TimeChart;
class DigitChart;
class DigitsClassifierController;
class QThread;
class QPushButton;
class QLabel;
class LearningConfigWidget;
namespace Neural { namespace Classifier { class TestResult; } }

class DigitsClassifierModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit DigitsClassifierModeWidget(QWidget* parent = nullptr);
    ~DigitsClassifierModeWidget();

    std::ostream* getTerminalStream();

    void setController(DigitsClassifierController* controller);
    void closeEvent(QCloseEvent* event) override;

public slots:
    void handleStatistics(const Neural::Classifier::TestResult& result);

private slots:
    void updateInfo();
    void handleOpenNetworkClicked();
    void handleOpenDatasetClicked();

private:
    std::unique_ptr<std::ostream> terminalStream;
    AdvancedTerminal* terminal = nullptr;
    TimeChart* chart = nullptr;
    DigitChart* digitChart = nullptr;
    QPushButton* toggleLearningButton = nullptr;
    QPushButton* openNetworkButton = nullptr;
    QPushButton* openTrainingDatasetButton = nullptr;
    QPushButton* openTestingDatasetButton = nullptr;
    QLabel* currentNetworkLabel = nullptr;
    QLabel* currentTestingDatasetLabel = nullptr;
    QLabel* currentTrainingDatasetLabel = nullptr;
    LearningConfigWidget* learningConfigWidget = nullptr;

    DigitsClassifierController* controller = nullptr;
};
