#pragma once

#include <QMainWindow>


class AdvancedTerminal;
class AdvancedTerminalStream;
class TimeChart;
class DigitChart;
namespace recognition {
class TestResult;
}
class DigitsRecognizerController;
class QThread;
class QPushButton;
class QLabel;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    AdvancedTerminal* getTerminalWidget() const;

    void setLogger(std::ostream* stream);
    void setController(DigitsRecognizerController* controller);

    void closeEvent(QCloseEvent* event) override;

public slots:
    void handleTimer();
    void handleStatistics(const recognition::TestResult& result);

private slots:
    void updateInfo();
    void handleOpenNetworkClicked();
    void handleOpenDatasetClicked();

private:

    std::ostream& logger();
    std::ostream* stream = nullptr;

    AdvancedTerminal* terminal = nullptr;
    TimeChart* chart = nullptr;
    DigitChart* digitChart = nullptr;
    QPushButton* toggleLearningButton = nullptr;
    QPushButton* openNetworkButton = nullptr;
    QPushButton* openDatasetButton = nullptr;
    QLabel* currentNetworkLabel = nullptr;
    QLabel* currentDatasetLabel = nullptr;

    DigitsRecognizerController* controller = nullptr;
};
