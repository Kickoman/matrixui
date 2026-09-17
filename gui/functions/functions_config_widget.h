#pragma once

#include "core/functions/config.h"

#include <QWidget>

#include <vector>

class QCheckBox;
class QDoubleSpinBox;
class QLineEdit;
class QPlainTextEdit;
class QSpinBox;

class FunctionsConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FunctionsConfigWidget(QWidget* parent = nullptr);

    void setConfig(const Genetizer::FunctionsConfig& config);
    Genetizer::FunctionsConfig getConfig() const;

signals:
    void edited();

private:
    Genetizer::FunctionsConfig stored;

    QSpinBox* maxPopulation = nullptr;
    QSpinBox* tournamentSize = nullptr;
    QDoubleSpinBox* populationDecrease = nullptr;
    QLineEdit* operators = nullptr;
    QDoubleSpinBox* scalarRange = nullptr;

    std::vector<QCheckBox*> functionChecks;

    QDoubleSpinBox* accuracyWeight = nullptr;
    QDoubleSpinBox* complexityWeight = nullptr;
    QDoubleSpinBox* lengthWeight = nullptr;
    QSpinBox* randomCount = nullptr;
    QSpinBox* randomDepth = nullptr;
    QSpinBox* seed = nullptr;
    QSpinBox* printTop = nullptr;

    QPlainTextEdit* initialExpressions = nullptr;
};
