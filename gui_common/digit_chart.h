#pragma once

#include <QWidget>

class QChart;
class QChartView;
class QBarSet;
class QBarCategoryAxis;
class QHorizontalBarSeries;
class QValueAxis;

class DigitChart : public QWidget
{
    Q_OBJECT
public:
    DigitChart(QWidget* parent = nullptr);

    QChart* getChart();
    unsigned getCount() const;

    void setTitle(const QString& title);
    void setValue(const unsigned digit, const double value);
    void setCount(unsigned count);

private:
    QChart* chart = nullptr;
    QChartView* view = nullptr;
    QHorizontalBarSeries* series = nullptr;
    QValueAxis* axisX = nullptr;
    QBarCategoryAxis* axisY = nullptr;
    QBarSet* barSet = nullptr;
};
