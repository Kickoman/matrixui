#include "gui_common/time_chart.h"

#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>
#include <QBoxLayout>

TimeChart::TimeChart(QWidget* parent)
    : QWidget(parent)
{
    chart = new QChart();
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->hide();

    chart->setTheme(QChart::ChartThemeQt);

    axisX = new QValueAxis();
    axisX->setRange(0, 20);
    axisY = new QValueAxis();
    axisY->setRange(0, 100);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);

    view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(view);

    startTime = QDateTime::currentDateTime();
}

void TimeChart::setTitle(const QString& title)
{
    chart->setTitle(title);
}

QValueAxis* TimeChart::getAxisX()
{
    return axisX;
}

QValueAxis* TimeChart::getAxisY()
{
    return axisY;
}

void TimeChart::addPoint(const double value, const QString& seriesName)
{
    const double timeOffset = startTime.msecsTo(QDateTime::currentDateTime()) / 1000.0;
    auto* series = getSeries(seriesName);
    series->append(timeOffset, value);

    const auto length = series->count();
    const auto first = series->at(std::max(0, length - 1 - 200));
    if (timeOffset > 20) {
        axisX->setRange(first.x(), timeOffset);
    } else {
        axisX->setRange(0, 20);
    };
}

QLineSeries* TimeChart::getSeries(const QString& name) {
    auto iter = allSeries.find(name);
    if (iter != allSeries.end()) {
        return iter->second;
    }

    auto* series = new QLineSeries();
    series->setName(name);
    chart->addSeries(series);
    series->setPointsVisible();
    series->attachAxis(axisX);
    series->attachAxis(axisY);
    allSeries[name] = series;

    if (allSeries.size() > 1) {
        chart->legend()->show();
    }

    return series;
}
