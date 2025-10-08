#pragma once

#include <QMainWindow>


class AdvancedTerminal;
class AdvancedTerminalStream;
class TimeChart;
class DigitChart;
class TestResult;
class DigitsRunner;
class QThread;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

public slots:
    void handleTimer();
    void handleStatistics(const TestResult& result);

private:
    void doDemo();
    std::ostream& logger();

    AdvancedTerminal* terminal = nullptr;
    std::ostream* stream = nullptr;
    TimeChart* chart = nullptr;
    DigitChart* digitChart = nullptr;
    DigitsRunner* runner = nullptr;
    QThread* thread = nullptr;
};
