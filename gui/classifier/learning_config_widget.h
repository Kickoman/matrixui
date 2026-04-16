#pragma once


#include <QWidget>
#include "core/lib/learning_config.h"

class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;

class LearningConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LearningConfigWidget(QWidget* parent = nullptr);

    void setConfig(const Neural::LearningConfig& config);
    Neural::LearningConfig getConfig() const;

private:
    QDoubleSpinBox* initialLearningRate = nullptr;
    QDoubleSpinBox* minLearningRate = nullptr;
    QDoubleSpinBox* learningRateDecay = nullptr;
    QSpinBox* maxEpochs = nullptr;
    QSpinBox* patience = nullptr;
    QSpinBox* innerEpochs = nullptr;
    QDoubleSpinBox* dropoutRate = nullptr;
    QSpinBox* datasetLimit = nullptr;
};
