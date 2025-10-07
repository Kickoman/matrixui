#pragma once

#include <QMainWindow>


class AdvancedTerminal;
class AdvancedTerminalStream;
class TimeChart;
class DigitChart;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

public slots:
    void handleTimer();

private:
    void doDemo();
    AdvancedTerminalStream& logger();

    AdvancedTerminal* terminal = nullptr;
    AdvancedTerminalStream* stream = nullptr;
    TimeChart* chart = nullptr;
    DigitChart* digitChart = nullptr;
};
