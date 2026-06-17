#include "gui/generator/gan_config_widget.h"

#include <QCheckBox>
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

    adaptiveLrCheck = new QCheckBox("Enable", this);

    lrEmaAlpha = new QDoubleSpinBox(this);
    lrEmaAlpha->setDecimals(3);
    lrEmaAlpha->setMinimum(0.0);
    lrEmaAlpha->setMaximum(0.999);
    lrEmaAlpha->setSingleStep(0.01);
    lrEmaAlpha->setToolTip("EMA smoothing: higher = slower reaction to score changes");

    lrAdjustFactor = new QDoubleSpinBox(this);
    lrAdjustFactor->setDecimals(3);
    lrAdjustFactor->setMinimum(1.001);
    lrAdjustFactor->setMaximum(2.0);
    lrAdjustFactor->setSingleStep(0.01);
    lrAdjustFactor->setToolTip("Multiplicative lr change per epoch when imbalance detected");

    lrWarmupEpochs = new QSpinBox(this);
    lrWarmupEpochs->setMinimum(0);
    lrWarmupEpochs->setMaximum(1000);
    lrWarmupEpochs->setToolTip("Epochs before adaptive adjustments begin");

    flatnessCheck = new QCheckBox("Enable", this);

    flatnessThreshold = new QDoubleSpinBox(this);
    flatnessThreshold->setDecimals(4);
    flatnessThreshold->setMinimum(0.0001);
    flatnessThreshold->setMaximum(0.1);
    flatnessThreshold->setSingleStep(0.001);
    flatnessThreshold->setToolTip("Max EMA change per epoch to count as flat");

    flatnessWindow = new QSpinBox(this);
    flatnessWindow->setMinimum(1);
    flatnessWindow->setMaximum(1000);
    flatnessWindow->setToolTip("Consecutive flat epochs before boosting both LRs");

    flatnessKickDuration = new QSpinBox(this);
    flatnessKickDuration->setMinimum(1);
    flatnessKickDuration->setMaximum(100);
    flatnessKickDuration->setToolTip("Epochs to hold the kick before reverting");

    flatnessDropoutBoost = new QDoubleSpinBox(this);
    flatnessDropoutBoost->setDecimals(2);
    flatnessDropoutBoost->setMinimum(1.0);
    flatnessDropoutBoost->setMaximum(10.0);
    flatnessDropoutBoost->setSingleStep(0.5);
    flatnessDropoutBoost->setToolTip("Multiply D dropout rate by this during kick (weakens D)");

    flatnessGenLrBoost = new QDoubleSpinBox(this);
    flatnessGenLrBoost->setDecimals(2);
    flatnessGenLrBoost->setMinimum(1.0);
    flatnessGenLrBoost->setMaximum(20.0);
    flatnessGenLrBoost->setSingleStep(0.5);
    flatnessGenLrBoost->setToolTip("Multiply G lr by this during kick (spikes G to exploit weakened D)");

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

    auto* adaptiveForm = new QFormLayout();
    adaptiveForm->addRow("Adaptive LR", adaptiveLrCheck);
    adaptiveForm->addRow("EMA alpha", lrEmaAlpha);
    adaptiveForm->addRow("Adjust factor", lrAdjustFactor);
    adaptiveForm->addRow("Warmup epochs", lrWarmupEpochs);

    auto* flatnessForm = new QFormLayout();
    flatnessForm->addRow("Flatness detect", flatnessCheck);
    flatnessForm->addRow("Threshold", flatnessThreshold);
    flatnessForm->addRow("Window", flatnessWindow);
    flatnessForm->addRow("Kick duration", flatnessKickDuration);
    flatnessForm->addRow("D dropout boost", flatnessDropoutBoost);
    flatnessForm->addRow("G lr boost", flatnessGenLrBoost);

    mainLayout->addLayout(leftForm);
    mainLayout->addLayout(rightForm);
    mainLayout->addLayout(adaptiveForm);
    mainLayout->addLayout(flatnessForm);

    setLayout(mainLayout);

    setConfig({});
}

void GanConfigWidget::setConfig(const Neural::GAN::LearningConfig& config) {
    epochs->setValue(static_cast<int>(config.epochs));
    batchSize->setValue(static_cast<int>(config.batchSize));
    generatorLr->setValue(config.generatorLearningRate);
    discriminatorLr->setValue(config.discriminatorLearningRate);
    discriminatorSteps->setValue(static_cast<int>(config.discriminatorStepsPerGenStep));
    dropoutRate->setValue(config.dropoutRate);
    classifierLossWeight->setValue(config.classifierLossWeight);
    latentDim->setValue(static_cast<int>(config.latentDim));
    datasetLimitPerLabel->setValue(static_cast<int>(config.datasetLimitPerLabel));
    adaptiveLrCheck->setChecked(config.adaptiveLr.enabled);
    lrEmaAlpha->setValue(config.adaptiveLr.lrEmaAlpha);
    lrAdjustFactor->setValue(config.adaptiveLr.lrAdjustFactor);
    lrWarmupEpochs->setValue(static_cast<int>(config.adaptiveLr.lrWarmupEpochs));
    flatnessCheck->setChecked(config.flatnessDetection.enabled);
    flatnessThreshold->setValue(config.flatnessDetection.threshold);
    flatnessWindow->setValue(static_cast<int>(config.flatnessDetection.window));
    flatnessKickDuration->setValue(static_cast<int>(config.flatnessDetection.kickDuration));
    flatnessDropoutBoost->setValue(config.flatnessDetection.discriminatorDropoutBoost);
    flatnessGenLrBoost->setValue(config.flatnessDetection.generatorLrBoost);
}

Neural::GAN::LearningConfig GanConfigWidget::getConfig() const {
    return {
        .latentDim = static_cast<std::size_t>(latentDim->value()),
        .generatorLearningRate = generatorLr->value(),
        .discriminatorLearningRate = discriminatorLr->value(),
        .epochs = static_cast<std::size_t>(epochs->value()),
        .batchSize = static_cast<std::size_t>(batchSize->value()),
        .discriminatorStepsPerGenStep = static_cast<std::size_t>(discriminatorSteps->value()),
        .dropoutRate = dropoutRate->value(),
        .classifierLossWeight = classifierLossWeight->value(),
        .datasetLimitPerLabel = static_cast<std::size_t>(datasetLimitPerLabel->value()),
        .adaptiveLr = {
            .enabled = adaptiveLrCheck->isChecked(),
            .lrEmaAlpha = lrEmaAlpha->value(),
            .lrAdjustFactor = lrAdjustFactor->value(),
            .lrWarmupEpochs = static_cast<std::size_t>(lrWarmupEpochs->value()),
        },
        .flatnessDetection = {
            .enabled = flatnessCheck->isChecked(),
            .threshold = flatnessThreshold->value(),
            .window = static_cast<std::size_t>(flatnessWindow->value()),
            .kickDuration = static_cast<std::size_t>(flatnessKickDuration->value()),
            .discriminatorDropoutBoost = flatnessDropoutBoost->value(),
            .generatorLrBoost = flatnessGenLrBoost->value(),
        }
    };
}
