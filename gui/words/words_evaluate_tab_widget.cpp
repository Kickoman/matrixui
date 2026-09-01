#include "gui/words/words_evaluate_tab_widget.h"

#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

WordsEvaluateTabWidget::WordsEvaluateTabWidget(QWidget* parent)
    : QWidget(parent)
{
    chooseAnalogiesButton = new QPushButton("Choose analogies file...", this);
    analogiesLabel = new QLabel("No analogies file", this);
    chooseSimilarityButton = new QPushButton("Choose similarity file...", this);
    similarityLabel = new QLabel("No similarity file", this);

    scoreColumnSpin = new QSpinBox(this);
    scoreColumnSpin->setRange(0, 100);
    restrictSpin = new QSpinBox(this);
    restrictSpin->setRange(0, 1'000'000);
    restrictSpin->setSpecialValueText("all");
    threadsSpin = new QSpinBox(this);
    threadsSpin->setRange(0, 256);
    threadsSpin->setSpecialValueText("auto");

    evaluateAnalogiesButton = new QPushButton("Evaluate analogies", this);
    evaluateSimilarityButton = new QPushButton("Evaluate similarity", this);

    auto* filesRow = new QHBoxLayout();
    filesRow->addWidget(chooseAnalogiesButton);
    filesRow->addWidget(analogiesLabel, 1);
    filesRow->addWidget(chooseSimilarityButton);
    filesRow->addWidget(similarityLabel, 1);

    auto* paramsForm = new QFormLayout();
    paramsForm->addRow("Score column", scoreColumnSpin);
    paramsForm->addRow("Restrict to", restrictSpin);
    paramsForm->addRow("Threads", threadsSpin);

    auto* actionRow = new QHBoxLayout();
    actionRow->addWidget(evaluateAnalogiesButton);
    actionRow->addWidget(evaluateSimilarityButton);
    actionRow->addStretch();

    auto* layout = new QVBoxLayout();
    layout->addLayout(filesRow);
    layout->addLayout(paramsForm);
    layout->addLayout(actionRow);
    layout->addStretch();
    setLayout(layout);
}

void WordsEvaluateTabWidget::setController(WordsController* newController)
{
    controller = newController;

    // Seeded once here, never in updateInfo -- setValue would re-fire
    // valueChanged and loop through infoUpdated.
    const auto info = controller->getInfo();
    scoreColumnSpin->setValue(static_cast<int>(info.scoreColumn));
    restrictSpin->setValue(static_cast<int>(info.restrictTo));
    threadsSpin->setValue(static_cast<int>(info.evaluateThreads));

    connect(chooseAnalogiesButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Choose analogies file", controller->getInfo().analogiesPath,
            "Analogy questions (*.txt);;All files (*)");
        if (!path.isEmpty()) {
            controller->setAnalogiesPath(path);
        }
    });
    connect(chooseSimilarityButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Choose similarity file", controller->getInfo().similarityPath,
            "Similarity dataset (*.txt *.tab *.csv);;All files (*)");
        if (!path.isEmpty()) {
            controller->setSimilarityPath(path);
        }
    });

    connect(evaluateAnalogiesButton, &QPushButton::clicked, [this] {
        controller->setEvaluateParams(
            static_cast<std::size_t>(scoreColumnSpin->value()),
            static_cast<std::size_t>(restrictSpin->value()),
            static_cast<std::size_t>(threadsSpin->value()));
        controller->evaluateAnalogies();
    });
    connect(evaluateSimilarityButton, &QPushButton::clicked, [this] {
        controller->setEvaluateParams(
            static_cast<std::size_t>(scoreColumnSpin->value()),
            static_cast<std::size_t>(restrictSpin->value()),
            static_cast<std::size_t>(threadsSpin->value()));
        controller->evaluateSimilarity();
    });
}

void WordsEvaluateTabWidget::updateInfo(const WordsController::Info& info)
{
    analogiesLabel->setText(
        info.analogiesPath.isEmpty() ? "No analogies file" : info.analogiesPath);
    similarityLabel->setText(
        info.similarityPath.isEmpty() ? "No similarity file" : info.similarityPath);

    const bool idle = !info.busy;
    chooseAnalogiesButton->setEnabled(idle);
    chooseSimilarityButton->setEnabled(idle);
    evaluateAnalogiesButton->setEnabled(idle && info.hasIndex && !info.analogiesPath.isEmpty());
    evaluateSimilarityButton->setEnabled(idle && info.hasIndex && !info.similarityPath.isEmpty());
}
