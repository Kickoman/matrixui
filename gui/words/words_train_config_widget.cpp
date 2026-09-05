#include "gui/words/words_train_config_widget.h"

#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QSpinBox>

WordsTrainConfigWidget::WordsTrainConfigWidget(QWidget* parent)
    : QWidget(parent)
{
    dimension = new QSpinBox(this);
    dimension->setRange(8, 1000);

    negatives = new QSpinBox(this);
    negatives->setRange(1, 64);

    window = new QSpinBox(this);
    window->setRange(1, 50);

    epochs = new QSpinBox(this);
    epochs->setRange(1, 100);

    sample = new QDoubleSpinBox(this);
    sample->setRange(0., 1.);
    sample->setDecimals(6);
    sample->setSingleStep(1e-4);

    learningRate = new QDoubleSpinBox(this);
    learningRate->setRange(0.0001, 1.);
    learningRate->setDecimals(4);
    learningRate->setSingleStep(0.005);

    threads = new QSpinBox(this);
    threads->setRange(0, 256);
    threads->setSpecialValueText("auto");

    auto* leftColumn = new QFormLayout();
    leftColumn->addRow("Dimension", dimension);
    leftColumn->addRow("Negatives", negatives);
    leftColumn->addRow("Window", window);
    leftColumn->addRow("Epochs", epochs);

    auto* rightColumn = new QFormLayout();
    rightColumn->addRow("Subsample", sample);
    rightColumn->addRow("Initial learning rate", learningRate);
    rightColumn->addRow("Threads", threads);

    auto* layout = new QHBoxLayout();
    layout->addLayout(leftColumn);
    layout->addLayout(rightColumn);
    setLayout(layout);

    setConfig(Words::WordsConfig{});
}

void WordsTrainConfigWidget::setConfig(const Words::WordsConfig& config)
{
    stored = config;
    dimension->setValue(static_cast<int>(config.model.dim));
    negatives->setValue(static_cast<int>(config.model.negatives));
    window->setValue(static_cast<int>(config.sampling.window));
    epochs->setValue(static_cast<int>(config.train.epochs));
    sample->setValue(config.sampling.sample);
    learningRate->setValue(config.model.initialLearningRate);
    threads->setValue(static_cast<int>(config.train.threads));
}

Words::WordsConfig WordsTrainConfigWidget::getConfig() const
{
    Words::WordsConfig result = stored;   // keep the fields this widget hides
    result.model.dim = static_cast<std::size_t>(dimension->value());
    result.model.negatives = static_cast<std::size_t>(negatives->value());
    result.sampling.window = static_cast<std::size_t>(window->value());
    result.train.epochs = static_cast<std::size_t>(epochs->value());
    result.sampling.sample = sample->value();
    result.model.initialLearningRate = learningRate->value();
    result.train.threads = static_cast<std::size_t>(threads->value());
    return result;
}
