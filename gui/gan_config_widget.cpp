#include "gan_config_widget.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>


GanConfigWidget::GanConfigWidget(QWidget* parent)
    : QWidget(parent)
{
    epochs = new QSpinBox(this);
    epochs->setMinimum(1);
    epochs->setMaximum(10000);

    batchSize = new QSpinBox(this);
    batchSize->setMinimum(1);
    batchSize->setMaximum(10000);

    generatorLr = new QDoubleSpinBox(this);
    generatorLr->setDecimals(6);
    generatorLr->setMinimum(0.0);
    generatorLr->setMaximum(1.0);
    generatorLr->setSingleStep(0.0001);

    discriminatorLr = new QDoubleSpinBox(this);
    discriminatorLr->setDecimals(6);
    discriminatorLr->setMinimum(0.0);
    discriminatorLr->setMaximum(1.0);
    discriminatorLr->setSingleStep(0.0001);

    discriminatorSteps = new QSpinBox(this);
    discriminatorSteps->setMinimum(1);
    discriminatorSteps->setMaximum(100);

    dropoutRate = new QDoubleSpinBox(this);
    dropoutRate->setDecimals(3);
    dropoutRate->setMinimum(0.0);
    dropoutRate->setMaximum(1.0);
    dropoutRate->setSingleStep(0.1);

    classifierLossWeight = new QDoubleSpinBox(this);
    classifierLossWeight->setDecimals(3);
    classifierLossWeight->setMinimum(0.0);
    classifierLossWeight->setMaximum(100.0);
    classifierLossWeight->setSingleStep(0.1);

    latentDim = new QSpinBox(this);
    latentDim->setMinimum(1);
    latentDim->setMaximum(1024);

    datasetLimitPerLabel = new QSpinBox(this);
    datasetLimitPerLabel->setMinimum(0);
    datasetLimitPerLabel->setMaximum(1000000);
    datasetLimitPerLabel->setSpecialValueText("No limit");

    auto* mainLayout = new QHBoxLayout(this);
    auto* leftForm = new QFormLayout();
    leftForm->addRow("Epochs", epochs);
    leftForm->addRow("Batch size", batchSize);
    leftForm->addRow("Generator LR", generatorLr);
    leftForm->addRow("Discriminator LR", discriminatorLr);

    auto* rightForm = new QFormLayout();
    rightForm->addRow("Disc steps / gen step", discriminatorSteps);
    rightForm->addRow("Dropout rate", dropoutRate);
    rightForm->addRow("Classifier loss weight", classifierLossWeight);
    rightForm->addRow("Latent dim", latentDim);
    rightForm->addRow("Dataset limit / label", datasetLimitPerLabel);

    mainLayout->addLayout(leftForm);
    mainLayout->addLayout(rightForm);

    setLayout(mainLayout);

    setConfig({});
}

void GanConfigWidget::setConfig(const Neural::GanConfig& config) {
    epochs->setValue(static_cast<int>(config.epochs));
    batchSize->setValue(static_cast<int>(config.batchSize));
    generatorLr->setValue(config.generatorLr);
    discriminatorLr->setValue(config.discriminatorLr);
    discriminatorSteps->setValue(static_cast<int>(config.discriminatorStepsPerGenStep));
    dropoutRate->setValue(config.dropoutRate);
    classifierLossWeight->setValue(config.classifierLossWeight);
    latentDim->setValue(static_cast<int>(config.latentDim));
    datasetLimitPerLabel->setValue(static_cast<int>(config.datasetLimitPerLabel));
}

Neural::GanConfig GanConfigWidget::getConfig() const {
    return {
        .latentDim = static_cast<std::size_t>(latentDim->value()),
        .numClasses = 10,
        .generatorLr = generatorLr->value(),
        .discriminatorLr = discriminatorLr->value(),
        .epochs = static_cast<std::size_t>(epochs->value()),
        .batchSize = static_cast<std::size_t>(batchSize->value()),
        .discriminatorStepsPerGenStep = static_cast<std::size_t>(discriminatorSteps->value()),
        .dropoutRate = dropoutRate->value(),
        .classifierLossWeight = classifierLossWeight->value(),
        .datasetLimitPerLabel = static_cast<std::size_t>(datasetLimitPerLabel->value()),
    };
}
