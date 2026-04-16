#include "gui/classifier/learning_config_widget.h"
#include "core/lib/learning_config.h"

#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <qspinbox.h>


LearningConfigWidget::LearningConfigWidget(QWidget* parent)
    : QWidget(parent)
{
    initialLearningRate = new QDoubleSpinBox(this);
    initialLearningRate->setDecimals(3);
    initialLearningRate->setMinimum(0);
    initialLearningRate->setMaximum(2);
    initialLearningRate->setSingleStep(0.25);

    minLearningRate = new QDoubleSpinBox(this);
    minLearningRate->setDecimals(3);
    minLearningRate->setMinimum(0);
    minLearningRate->setMaximum(2);
    minLearningRate->setSingleStep(0.25);

    learningRateDecay = new QDoubleSpinBox(this);
    learningRateDecay->setDecimals(3);
    learningRateDecay->setMinimum(0);
    learningRateDecay->setMaximum(1);
    learningRateDecay->setSingleStep(0.1);

    maxEpochs = new QSpinBox(this);
    maxEpochs->setMinimum(1);
    maxEpochs->setMaximum(1000);

    patience = new QSpinBox(this);
    patience->setMinimum(2);
    patience->setMaximum(1000);

    innerEpochs = new QSpinBox(this);
    innerEpochs->setMinimum(1);
    innerEpochs->setMaximum(1000);

    dropoutRate = new QDoubleSpinBox(this);
    dropoutRate->setDecimals(3);
    dropoutRate->setMinimum(0);
    dropoutRate->setMaximum(1);
    dropoutRate->setSingleStep(0.1);

    datasetLimit = new QSpinBox(this);
    datasetLimit->setMinimum(1);
    datasetLimit->setMaximum(1000000);

    auto* mainLayout = new QHBoxLayout(this);
    auto* leftFormLayout = new QFormLayout(this);
    leftFormLayout->addRow("Initial LR", initialLearningRate);
    leftFormLayout->addRow("LR decay", learningRateDecay);
    leftFormLayout->addRow("Patience", patience);
    leftFormLayout->addRow("Dropout rate", dropoutRate);

    auto* rightFormLayout = new QFormLayout(this);
    rightFormLayout->addRow("Minimum LR", minLearningRate);
    rightFormLayout->addRow("Max epochs", maxEpochs);
    rightFormLayout->addRow("Inner epochs", innerEpochs);
    rightFormLayout->addRow("Dataset file limit", datasetLimit);

    mainLayout->addLayout(leftFormLayout);
    mainLayout->addLayout(rightFormLayout);

    setLayout(mainLayout);
}

void LearningConfigWidget::setConfig(const Neural::LearningConfig& config) {
    initialLearningRate->setValue(config.initialLearningRate);
    minLearningRate->setValue(config.minLearningRate);
    learningRateDecay->setValue(config.learningRateDecay);
    maxEpochs->setValue(config.maxEpochs);
    patience->setValue(config.patience);
    innerEpochs->setValue(config.innerEpochs);
    dropoutRate->setValue(config.dropoutRate);
    datasetLimit->setValue(config.datasetLimitPerLabel);
}

Neural::LearningConfig LearningConfigWidget::getConfig() const {
    return {
        .initialLearningRate = initialLearningRate->value(),
        .minLearningRate = minLearningRate->value(),
        .learningRateDecay = learningRateDecay->value(),
        .maxEpochs = static_cast<std::size_t>(maxEpochs->value()),
        .patience = static_cast<std::size_t>(patience->value()),
        .innerEpochs = static_cast<std::size_t>(innerEpochs->value()),
        .datasetLimitPerLabel = static_cast<std::size_t>(datasetLimit->value()),
        .dropoutRate = dropoutRate->value(),
    };
}
