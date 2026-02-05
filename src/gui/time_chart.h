#pragma once

#include <QWidget>
#include <QDateTime>

class QChart;
class QLineSeries;
class QChartView;
class QValueAxis;

class TimeChart : public QWidget
{
    Q_OBJECT
public:
    TimeChart(QWidget* parent = nullptr);

    void setTitle(const QString& title);
    QValueAxis* getAxisX();
    QValueAxis* getAxisY();
    void addPoint(const double value);

private:
    void updateYAxisRange();

private:
    QChart* chart = nullptr;
    QLineSeries* series = nullptr;
    QChartView* view = nullptr;
    QValueAxis* axisX = nullptr;
    QValueAxis* axisY = nullptr;
    QDateTime startTime;
};
