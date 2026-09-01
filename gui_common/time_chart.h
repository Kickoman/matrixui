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

    // When enabled, the Y axis grows to fit every value seen so far (with a
    // small margin) instead of staying at the fixed 0..100 percent range.
    void setAutoScaleY(bool enabled);
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

    bool autoScaleY = false;
    bool hasObservedValue = false;
    double observedMin = 0.;
    double observedMax = 0.;
};
