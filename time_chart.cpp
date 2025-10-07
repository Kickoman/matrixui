#include "time_chart.h"

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

    series = new QLineSeries();
    chart->addSeries(series);
    chart->setTheme(QChart::ChartThemeQt);
    series->setPointsVisible(true);

    axisX = new QValueAxis();
    axisX->setRange(0, 20);
    axisY = new QValueAxis();
    axisY->setRange(0, 100);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisX);
    series->attachAxis(axisY);

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

void TimeChart::addPoint(const double value)
{
    const double timeOffset = startTime.msecsTo(QDateTime::currentDateTime()) / 1000.0;
    series->append(timeOffset, value);
    // if (series->count() > 20) {
    //     series->remove(0);
    // }
    if (timeOffset > 20) {
        axisX->setRange(timeOffset - 20, timeOffset);
    } else {
        axisX->setRange(0, 20);
    }
    // updateYAxisRange();
}

void TimeChart::updateYAxisRange()
{
    if (series->count() == 0) return;

    double minY = 100, maxY = 0;
    for (int i = 0; i < series->count(); ++i) {
        QPointF point = series->at(i);
        if (point.y() < minY) minY = point.y();
        if (point.y() > maxY) maxY = point.y();
    }

    double padding = (maxY - minY) * 0.1;
    if (padding == 0) padding = 5;

    axisY->setRange(minY - padding, maxY + padding);
}
