#pragma once

#include <QWidget>
#include "core/generator/learning_config.h"

class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;

class GanConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit GanConfigWidget(QWidget* parent = nullptr);

    void setConfig(const Neural::GAN::LearningConfig& config);
    Neural::GAN::LearningConfig getConfig() const;

private:
    QSpinBox* epochs = nullptr;
    QSpinBox* batchSize = nullptr;
    QDoubleSpinBox* generatorLr = nullptr;
    QDoubleSpinBox* discriminatorLr = nullptr;
    QSpinBox* discriminatorSteps = nullptr;
    QDoubleSpinBox* dropoutRate = nullptr;
    QDoubleSpinBox* classifierLossWeight = nullptr;
    QSpinBox* latentDim = nullptr;
    QSpinBox* datasetLimitPerLabel = nullptr;

    // Adaptive lr controls
    QCheckBox*      adaptiveLrCheck   = nullptr;
    QDoubleSpinBox* lrEmaAlpha        = nullptr;
    QDoubleSpinBox* lrAdjustFactor    = nullptr;
    QSpinBox*       lrWarmupEpochs    = nullptr;

    // Flatness detection controls
    QCheckBox*      flatnessCheck        = nullptr;
    QDoubleSpinBox* flatnessThreshold    = nullptr;
    QSpinBox*       flatnessWindow       = nullptr;
    QSpinBox*       flatnessKickDuration = nullptr;
    QDoubleSpinBox* flatnessDropoutBoost = nullptr;
    QDoubleSpinBox* flatnessGenLrBoost   = nullptr;
};
