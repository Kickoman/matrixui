#include "gui/functions/functions_points_model.h"

#include <algorithm>


FunctionsPointsModel::FunctionsPointsModel(QObject* parent)
    : QObject(parent)
{}

const QStringList& FunctionsPointsModel::getVariables() const {
    return variables;
}

void FunctionsPointsModel::setVariables(const QStringList& names) {
    if (names.isEmpty() || names == variables) {
        return;
    }

    for (auto& row : rows) {
        std::vector<double> rebuilt(names.size(), 0.0);
        for (int column = 0; column < names.size(); ++column) {
            const auto previous = variables.indexOf(names[column]);
            if (previous >= 0 && previous < static_cast<int>(row.values.size())) {
                rebuilt[column] = row.values[previous];
            }
        }
        row.values = std::move(rebuilt);
    }

    variables = names;
    emit changed();
}

int FunctionsPointsModel::getRowCount() const {
    return rows.size();
}

double FunctionsPointsModel::getValue(const int row, const int column) const {
    if (row < 0 || row >= rows.size()) {
        return 0;
    }
    if (column == variables.size()) {
        return rows[row].expected;
    }
    if (column < 0 || column >= static_cast<int>(rows[row].values.size())) {
        return 0;
    }
    return rows[row].values[column];
}

void FunctionsPointsModel::setValue(const int row, const int column, const double value) {
    if (row < 0 || row >= rows.size()) {
        return;
    }
    if (column == variables.size()) {
        rows[row].expected = value;
    } else if (column >= 0 && column < static_cast<int>(rows[row].values.size())) {
        rows[row].values[column] = value;
    } else {
        return;
    }
    emit changed();
}

void FunctionsPointsModel::addRow() {
    rows.push_back(Row{std::vector<double>(variables.size(), 0.0), 0.0});
    emit changed();
}

void FunctionsPointsModel::removeRow(const int row) {
    if (row < 0 || row >= rows.size()) {
        return;
    }
    rows.remove(row);
    emit changed();
}

void FunctionsPointsModel::clear() {
    if (rows.isEmpty()) {
        return;
    }
    rows.clear();
    emit changed();
}

std::vector<Genetizer::Entry> FunctionsPointsModel::toEntries() const {
    std::vector<Genetizer::Entry> entries;
    entries.reserve(rows.size());
    for (const auto& row : rows) {
        Genetizer::Entry entry;
        entry.variables.reserve(variables.size());
        for (int column = 0; column < variables.size(); ++column) {
            entry.variables.push_back(Genetizer::Variable{
                .name = variables[column].toStdString(),
                .value = column < static_cast<int>(row.values.size()) ? row.values[column] : 0.0,
            });
        }
        entry.expectedResult = row.expected;
        entries.push_back(std::move(entry));
    }
    return entries;
}

void FunctionsPointsModel::setFromEntries(const std::vector<Genetizer::Entry>& entries) {
    QStringList names;
    if (!entries.empty()) {
        for (const auto& variable : entries.front().variables) {
            names.append(QString::fromStdString(variable.name));
        }
    }
    if (names.isEmpty()) {
        names = variables;
    }

    QVector<Row> rebuilt;
    rebuilt.reserve(static_cast<int>(entries.size()));
    for (const auto& entry : entries) {
        Row row;
        row.values.assign(names.size(), 0.0);
        for (int column = 0; column < names.size(); ++column) {
            const auto found = std::find_if(
                entry.variables.cbegin(), entry.variables.cend(),
                [&names, column](const Genetizer::Variable& variable) {
                    return variable.name == names[column].toStdString();
                });
            if (found != entry.variables.cend()) {
                row.values[column] = found->value;
            }
        }
        row.expected = entry.expectedResult;
        rebuilt.push_back(std::move(row));
    }

    variables = names;
    rows = std::move(rebuilt);
    emit changed();
}

QVector<QPointF> FunctionsPointsModel::toPlotPoints() const {
    QVector<QPointF> result;
    if (variables.size() != 1) {
        return result;
    }
    result.reserve(rows.size());
    for (const auto& row : rows) {
        result.append(QPointF(row.values.empty() ? 0.0 : row.values.front(), row.expected));
    }
    return result;
}

void FunctionsPointsModel::addPoint(const double x, const double expected) {
    if (variables.size() != 1) {
        return;
    }
    rows.push_back(Row{{x}, expected});
    emit changed();
}

void FunctionsPointsModel::movePoint(const int row, const double x, const double expected) {
    if (variables.size() != 1 || row < 0 || row >= rows.size()) {
        return;
    }
    rows[row].values.assign(1, x);
    rows[row].expected = expected;
    emit changed();
}
