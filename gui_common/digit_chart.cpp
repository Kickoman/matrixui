#include "gui_common/digit_chart.h"

#include <QChart>
#include <QChartView>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <QHorizontalBarSeries>
#include <QVBoxLayout>
#include <QValueAxis>
#include <qobject.h>


DigitChart::DigitChart(QWidget* parent)
    : QWidget(parent)
{
    chart = new QChart();
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->hide();

    axisY = new QBarCategoryAxis();
    axisY->append(
        {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    );

    barSet = new QBarSet("");
    *barSet << 0 << 1 << 2 << 0 << 0 << 0 << 0 << 0 << 0 << 0;
    series = new QHorizontalBarSeries();
    series->append(barSet);

    chart->addSeries(series);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    axisX = new QValueAxis();
    axisX->setRange(0, 100);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);


    view = new QChartView(chart);
    auto* layout = new QVBoxLayout(this);
    layout->addWidget(view);
}

QChart* DigitChart::getChart() {
    return chart;
}

void DigitChart::setTitle(const QString& title) {
    chart->setTitle(title);
}

void DigitChart::setValue(const unsigned digit, const double value) {
    barSet->replace(digit, value);
}

void DigitChart::setCount(const unsigned count) {
    axisY->clear();
    for (unsigned i = 0; i < count; ++i) {
        axisY->append(QString::number(i));
    }
    if (barSet->count() < count) {
        for (unsigned i = 0; i < count - barSet->count(); ++i) {
            *barSet << 0;
        }
    }
}

unsigned DigitChart::getCount() const {
    return axisY->count();
}
