#pragma once

#include "core/functions/applier.h"

#include <QObject>
#include <QPointF>
#include <QStringList>
#include <QVector>

#include <vector>


class FunctionsPointsModel : public QObject
{
    Q_OBJECT
public:
    explicit FunctionsPointsModel(QObject* parent = nullptr);

    const QStringList& getVariables() const;
    void setVariables(const QStringList& names);

    int getRowCount() const;
    double getValue(int row, int column) const;   // column == variables.size() is expected
    void setValue(int row, int column, double value);

    void addRow();
    void removeRow(int row);
    void clear();

    std::vector<Genetizer::Entry> toEntries() const;
    void setFromEntries(const std::vector<Genetizer::Entry>& entries);

    QVector<QPointF> toPlotPoints() const;
    void addPoint(double x, double expected);
    void movePoint(int row, double x, double expected);

signals:
    void changed();
    void rowChanged(int row);

private:
    struct Row {
        std::vector<double> values;
        double expected = 0;
    };

    QStringList variables{"x"};
    QVector<Row> rows;
};
