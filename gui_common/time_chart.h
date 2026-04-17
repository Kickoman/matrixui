#pragma once

#include <QWidget>
#include <QDateTime>
#include <unordered_map>

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
    QChart* getChart();
    QValueAxis* getAxisX();
    QValueAxis* getAxisY();
    void addPoint(double value, const QString& seriesName = {});

private:
    QLineSeries* getSeries(const QString& name);

private:
    QChart* chart = nullptr;
    std::unordered_map<QString /*series name*/, QLineSeries*> allSeries;
    QChartView* view = nullptr;
    QValueAxis* axisX = nullptr;
    QValueAxis* axisY = nullptr;
    QDateTime startTime;
};
