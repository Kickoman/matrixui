#pragma once

#include "gui/words/words_controller.h"

#include <QWidget>

class TimeChart;
class WordsTrainConfigWidget;
class QComboBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QSpinBox;

class WordsDataTabWidget : public QWidget
{
    Q_OBJECT
public:
    explicit WordsDataTabWidget(QWidget* parent = nullptr);

    void setController(WordsController* controller);
    void updateInfo(const WordsController::Info& info);

public slots:
    void handleTrainProgress(const Words::TrainProgress& progress);

private:
    WordsController* controller = nullptr;

    QPushButton* openDumpButton = nullptr;
    QPushButton* inspectButton = nullptr;
    QPushButton* buildVocabularyButton = nullptr;
    QPushButton* loadVocabularyButton = nullptr;
    QPushButton* buildCorpusButton = nullptr;
    QPushButton* loadCorpusButton = nullptr;

    QLabel* dumpLabel = nullptr;
    QLabel* vocabularyLabel = nullptr;
    QLabel* corpusLabel = nullptr;
    QLabel* embeddingsLabel = nullptr;
    QSpinBox* minCountSpin = nullptr;
    QComboBox* corpusStorageCombo = nullptr;
    WordsTrainConfigWidget* configWidget = nullptr;

    Words::CorpusStorage selectedCorpusStorage() const;

    QPushButton* toggleTrainingButton = nullptr;
    QPushButton* loadEmbeddingsButton = nullptr;
    QPushButton* saveEmbeddingsButton = nullptr;
    QProgressBar* progressBar = nullptr;
    QLabel* etaLabel = nullptr;
    QLabel* statusLabel = nullptr;

    TimeChart* lossChart = nullptr;
    TimeChart* speedChart = nullptr;
};
