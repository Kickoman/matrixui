#pragma once

#include "core/words/config.h"

#include <QWidget>

class QDoubleSpinBox;
class QSpinBox;

// Editor for the training knobs the CLI exposes. Fields not shown here
// (chunk size, seed, probe pairs, ...) round-trip through getConfig()
// untouched, so they stay editable by hand in the settings file.
class WordsTrainConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit WordsTrainConfigWidget(QWidget* parent = nullptr);

    void setConfig(const Words::WordsConfig& config);
    Words::WordsConfig getConfig() const;

private:
    Words::WordsConfig stored;   // carries the hidden fields

    QSpinBox* dimension = nullptr;
    QSpinBox* negatives = nullptr;
    QSpinBox* window = nullptr;
    QSpinBox* epochs = nullptr;
    QDoubleSpinBox* sample = nullptr;
    QDoubleSpinBox* learningRate = nullptr;
    QSpinBox* threads = nullptr;
    QSpinBox* buckets = nullptr;
    QSpinBox* minN = nullptr;
    QSpinBox* maxN = nullptr;
};
