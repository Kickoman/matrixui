#include "gui/words/words_explore_tab_widget.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

WordsExploreTabWidget::WordsExploreTabWidget(QWidget* parent)
    : QWidget(parent)
{
    neighboursWordEdit = new QLineEdit(this);
    neighboursWordEdit->setPlaceholderText("king");
    neighboursCountSpin = new QSpinBox(this);
    neighboursCountSpin->setRange(1, 100);
    neighboursCountSpin->setValue(10);
    neighboursButton = new QPushButton("Neighbours", this);
    batteryButton = new QPushButton("Run default battery", this);

    expressionEdit = new QLineEdit(this);
    expressionEdit->setPlaceholderText("king - man + woman");
    expressionCountSpin = new QSpinBox(this);
    expressionCountSpin->setRange(1, 100);
    expressionCountSpin->setValue(10);
    expressionButton = new QPushButton("Evaluate expression", this);

    oddOneEdit = new QLineEdit(this);
    oddOneEdit->setPlaceholderText("breakfast cereal lunch dinner");
    oddOneButton = new QPushButton("Find odd one out", this);

    axisEdit = new QLineEdit(this);
    axisEdit->setPlaceholderText("good - bad");
    axisWordsEdit = new QLineEdit(this);
    axisWordsEdit->setPlaceholderText("optional: words to project");
    axisRestrictSpin = new QSpinBox(this);
    axisRestrictSpin->setRange(0, 1'000'000);
    axisRestrictSpin->setValue(30000);
    axisRestrictSpin->setSpecialValueText("all");
    axisCountSpin = new QSpinBox(this);
    axisCountSpin->setRange(1, 100);
    axisCountSpin->setValue(10);
    axisButton = new QPushButton("Project onto axis", this);

    auto* neighboursForm = new QFormLayout();
    neighboursForm->addRow("Word", neighboursWordEdit);
    neighboursForm->addRow("Count", neighboursCountSpin);
    neighboursForm->addRow(neighboursButton);
    neighboursForm->addRow(batteryButton);

    auto* expressionForm = new QFormLayout();
    expressionForm->addRow("Expression", expressionEdit);
    expressionForm->addRow("Count", expressionCountSpin);
    expressionForm->addRow(expressionButton);

    auto* oddOneForm = new QFormLayout();
    oddOneForm->addRow("Words", oddOneEdit);
    oddOneForm->addRow(oddOneButton);

    auto* axisForm = new QFormLayout();
    axisForm->addRow("Axis", axisEdit);
    axisForm->addRow("Words", axisWordsEdit);
    axisForm->addRow("Restrict to", axisRestrictSpin);
    axisForm->addRow("Count", axisCountSpin);
    axisForm->addRow(axisButton);

    auto* leftColumn = new QVBoxLayout();
    leftColumn->addLayout(neighboursForm);
    leftColumn->addLayout(expressionForm);
    leftColumn->addStretch();

    auto* rightColumn = new QVBoxLayout();
    rightColumn->addLayout(oddOneForm);
    rightColumn->addLayout(axisForm);
    rightColumn->addStretch();

    auto* layout = new QHBoxLayout();
    layout->addLayout(leftColumn);
    layout->addLayout(rightColumn);
    setLayout(layout);
}

void WordsExploreTabWidget::setController(WordsController* newController)
{
    controller = newController;

    connect(neighboursButton, &QPushButton::clicked, [this] {
        controller->queryNeighbours(neighboursWordEdit->text(), neighboursCountSpin->value());
    });
    connect(batteryButton, &QPushButton::clicked, [this] {
        controller->runDefaultBattery();
    });
    connect(expressionButton, &QPushButton::clicked, [this] {
        controller->queryExpression(expressionEdit->text(), expressionCountSpin->value());
    });
    connect(oddOneButton, &QPushButton::clicked, [this] {
        controller->queryOddOneOut(oddOneEdit->text());
    });
    connect(axisButton, &QPushButton::clicked, [this] {
        controller->queryAxis(axisEdit->text(), axisWordsEdit->text(),
                              axisRestrictSpin->value(), axisCountSpin->value());
    });

    connect(neighboursWordEdit, &QLineEdit::returnPressed, neighboursButton, &QPushButton::click);
    connect(expressionEdit, &QLineEdit::returnPressed, expressionButton, &QPushButton::click);
    connect(oddOneEdit, &QLineEdit::returnPressed, oddOneButton, &QPushButton::click);
    connect(axisEdit, &QLineEdit::returnPressed, axisButton, &QPushButton::click);
}

void WordsExploreTabWidget::updateInfo(const WordsController::Info& info)
{
    // Deliberately not gated on info.busy: the index is an immutable snapshot
    // only swapped on the GUI thread, so querying while a background task runs
    // is coherent.
    const bool ready = info.hasIndex;
    for (auto* widget : std::initializer_list<QWidget*>{
             neighboursWordEdit, neighboursCountSpin, neighboursButton, batteryButton,
             expressionEdit, expressionCountSpin, expressionButton,
             oddOneEdit, oddOneButton,
             axisEdit, axisWordsEdit, axisRestrictSpin, axisCountSpin, axisButton}) {
        widget->setEnabled(ready);
    }
}
