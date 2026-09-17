#include "gui/functions/functions_config_widget.h"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QSpinBox>
#include <QVBoxLayout>

#include <string>
#include <unordered_set>


FunctionsConfigWidget::FunctionsConfigWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    auto* columns = new QHBoxLayout();
    auto* left = new QFormLayout();
    auto* right = new QFormLayout();
    columns->addLayout(left);
    columns->addLayout(right);
    root->addLayout(columns);

    maxPopulation = new QSpinBox(this);
    maxPopulation->setRange(1, 10'000'000);
    left->addRow("Population", maxPopulation);

    tournamentSize = new QSpinBox(this);
    tournamentSize->setRange(1, 64);
    left->addRow("Tournament", tournamentSize);

    populationDecrease = new QDoubleSpinBox(this);
    populationDecrease->setRange(0, 1);
    populationDecrease->setDecimals(3);
    populationDecrease->setSingleStep(0.05);
    left->addRow("Survivor share", populationDecrease);

    operators = new QLineEdit(this);
    operators->setToolTip("Binary operators available to mutation, e.g. +-*/^");
    left->addRow("Operators", operators);

    scalarRange = new QDoubleSpinBox(this);
    scalarRange->setRange(0, 1'000'000);
    scalarRange->setDecimals(2);
    left->addRow("Scalar range", scalarRange);

    const auto makeWeight = [this](QFormLayout* form, const QString& label) {
        auto* box = new QDoubleSpinBox(this);
        box->setRange(0, 1000);
        box->setDecimals(4);
        box->setSingleStep(0.01);
        form->addRow(label, box);
        return box;
    };
    accuracyWeight = makeWeight(right, "Accuracy weight");
    complexityWeight = makeWeight(right, "Complexity weight");
    lengthWeight = makeWeight(right, "Length weight");

    randomCount = new QSpinBox(this);
    randomCount->setRange(0, 1'000'000);
    right->addRow("Random organisms", randomCount);

    randomDepth = new QSpinBox(this);
    randomDepth->setRange(0, 20);
    right->addRow("Random depth", randomDepth);

    seed = new QSpinBox(this);
    seed->setRange(0, 2'000'000'000);
    seed->setSpecialValueText("random");
    right->addRow("Seed", seed);

    printTop = new QSpinBox(this);
    printTop->setRange(0, 1000);
    printTop->setSpecialValueText("all");
    right->addRow("Rows logged", printTop);

    root->addWidget(new QLabel("Functions available to mutation", this));
    auto* functionsGrid = new QGridLayout();
    int functionRow = 0;
    int functionColumn = 0;
    for (const auto& holder : Matematyka::rpn::FUNCTIONS_AVAILABLE<double>) {
        auto* box = new QCheckBox(
            QString::fromUtf8(holder.name.data(), static_cast<int>(holder.name.size())), this);
        box->setToolTip(
            "Unchecking only stops mutation and random growth from introducing it.\n"
            "An initial expression may still use it, and such an organism keeps working.");
        functionsGrid->addWidget(box, functionRow, functionColumn);
        functionChecks.push_back(box);
        if (++functionColumn == 4) {
            functionColumn = 0;
            ++functionRow;
        }
    }
    root->addLayout(functionsGrid);

    root->addWidget(new QLabel("Initial expressions (one per line)", this));
    initialExpressions = new QPlainTextEdit(this);
    initialExpressions->setMaximumHeight(90);
    initialExpressions->setToolTip(
        "Seeded expressions are the only source of exact constants: random ones are drawn from a\n"
        "continuous range, so a constant of exactly 2 never appears on its own. Seed x/2 or x+1\n"
        "when the answer is expected to be a rational formula.");
    root->addWidget(initialExpressions);

    setConfig({});

    for (auto* box : {maxPopulation, tournamentSize, randomCount, randomDepth, seed, printTop}) {
        connect(box, &QSpinBox::valueChanged, this, &FunctionsConfigWidget::edited);
    }
    for (auto* box : {populationDecrease, scalarRange, accuracyWeight, complexityWeight, lengthWeight}) {
        connect(box, &QDoubleSpinBox::valueChanged, this, &FunctionsConfigWidget::edited);
    }
    connect(operators, &QLineEdit::textChanged, this, &FunctionsConfigWidget::edited);
    for (auto* box : functionChecks) {
        connect(box, &QCheckBox::toggled, this, &FunctionsConfigWidget::edited);
    }
    connect(initialExpressions, &QPlainTextEdit::textChanged, this, &FunctionsConfigWidget::edited);
}

void FunctionsConfigWidget::setConfig(const Genetizer::FunctionsConfig& config) {
    stored = config;

    maxPopulation->setValue(static_cast<int>(config.genetizer.maxPopulation));
    tournamentSize->setValue(static_cast<int>(config.genetizer.tournamentSize));
    populationDecrease->setValue(config.genetizer.populationDecreaseFactor);
    operators->setText(QString::fromStdString(config.mutation.operators));
    scalarRange->setValue(config.mutation.scalarRange);

    const std::unordered_set<std::string> enabled(
        config.mutation.functions.begin(), config.mutation.functions.end());
    for (std::size_t i = 0; i < functionChecks.size(); ++i) {
        functionChecks[i]->setChecked(
            enabled.contains(std::string(Matematyka::rpn::FUNCTIONS_AVAILABLE<double>[i].name)));
    }

    accuracyWeight->setValue(config.fitness.accuracyWeight);
    complexityWeight->setValue(config.fitness.complexityWeight);
    lengthWeight->setValue(config.fitness.lengthWeight);
    randomCount->setValue(static_cast<int>(config.randomCount));
    randomDepth->setValue(static_cast<int>(config.randomDepth));
    seed->setValue(static_cast<int>(config.seed));
    printTop->setValue(static_cast<int>(config.printTop));

    QStringList expressions;
    for (const auto& expression : config.initialExpressions) {
        expressions.append(QString::fromStdString(expression));
    }
    initialExpressions->setPlainText(expressions.join('\n'));
}

Genetizer::FunctionsConfig FunctionsConfigWidget::getConfig() const {
    auto result = stored;   // keep the fields this widget hides

    result.genetizer.maxPopulation = static_cast<std::size_t>(maxPopulation->value());
    result.genetizer.tournamentSize = static_cast<std::size_t>(tournamentSize->value());
    result.genetizer.populationDecreaseFactor = populationDecrease->value();
    result.mutation.operators = operators->text().toStdString();
    result.mutation.scalarRange = scalarRange->value();

    result.mutation.functions.clear();
    for (std::size_t i = 0; i < functionChecks.size(); ++i) {
        if (functionChecks[i]->isChecked()) {
            result.mutation.functions.emplace_back(
                Matematyka::rpn::FUNCTIONS_AVAILABLE<double>[i].name);
        }
    }

    result.fitness.accuracyWeight = accuracyWeight->value();
    result.fitness.complexityWeight = complexityWeight->value();
    result.fitness.lengthWeight = lengthWeight->value();
    result.randomCount = static_cast<std::size_t>(randomCount->value());
    result.randomDepth = static_cast<std::size_t>(randomDepth->value());
    result.seed = static_cast<std::uint64_t>(seed->value());
    result.printTop = static_cast<std::size_t>(printTop->value());

    result.initialExpressions.clear();
    for (const auto& line : initialExpressions->toPlainText().split('\n')) {
        const auto trimmed = line.trimmed();
        if (!trimmed.isEmpty()) {
            result.initialExpressions.push_back(trimmed.toStdString());
        }
    }
    return result;
}
