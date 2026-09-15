#include "gui/words/words_data_tab_widget.h"

#include "gui/lib/theme.h"
#include "gui/words/words_train_config_widget.h"
#include "gui_common/time_chart.h"

#include <QComboBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

WordsDataTabWidget::WordsDataTabWidget(QWidget* parent)
    : QWidget(parent)
{
    openDumpButton = new QPushButton("Open text dump...", this);
    inspectButton = new QPushButton("Inspect dump", this);
    buildVocabularyButton = new QPushButton("Build vocabulary...", this);
    loadVocabularyButton = new QPushButton("Load vocabulary...", this);
    buildCorpusButton = new QPushButton("Build corpus...", this);
    loadCorpusButton = new QPushButton("Load corpus...", this);

    dumpLabel = new QLabel("No text dump", this);
    vocabularyLabel = new QLabel("No vocabulary", this);
    corpusLabel = new QLabel("No corpus", this);
    embeddingsLabel = new QLabel("No embeddings", this);

    minCountSpin = new QSpinBox(this);
    minCountSpin->setRange(1, 1000);
    minCountSpin->setValue(5);

    corpusStorageCombo = new QComboBox(this);
    corpusStorageCombo->addItem("auto", static_cast<int>(Words::CorpusStorage::Auto));
    corpusStorageCombo->addItem("mmap", static_cast<int>(Words::CorpusStorage::Mapped));
    corpusStorageCombo->addItem("load", static_cast<int>(Words::CorpusStorage::Loaded));
    corpusStorageCombo->setToolTip(
        "How the corpus is held: load reads it into memory, mmap maps the file "
        "read-only, auto loads only small files");

    configWidget = new WordsTrainConfigWidget(this);

    toggleTrainingButton = new QPushButton("Start training", this);
    loadEmbeddingsButton = new QPushButton("Load embeddings...", this);
    saveEmbeddingsButton = new QPushButton("Save embeddings...", this);

    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 1000);
    progressBar->setValue(0);
    etaLabel = new QLabel(this);
    statusLabel = new QLabel(this);

    lossChart = new TimeChart(this);
    lossChart->setTitle("Probe loss");
    lossChart->setAutoScaleY(true);
    AppTheme::ApplyTheme(lossChart->getChart());

    speedChart = new TimeChart(this);
    speedChart->setTitle("Training speed (Mpairs/s)");
    speedChart->setAutoScaleY(true);
    AppTheme::ApplyTheme(speedChart->getChart());

    auto* root = new QVBoxLayout();
    auto* dataButtonBar = new QHBoxLayout();
    auto* infoRow = new QHBoxLayout();
    auto* infoColumn = new QVBoxLayout();
    auto* minCountForm = new QFormLayout();
    auto* trainingBar = new QHBoxLayout();
    auto* chartRow = new QHBoxLayout();

    dataButtonBar->addWidget(openDumpButton);
    dataButtonBar->addWidget(inspectButton);
    dataButtonBar->addWidget(buildVocabularyButton);
    dataButtonBar->addWidget(loadVocabularyButton);
    dataButtonBar->addWidget(buildCorpusButton);
    dataButtonBar->addWidget(loadCorpusButton);

    infoColumn->addWidget(dumpLabel);
    infoColumn->addWidget(vocabularyLabel);
    infoColumn->addWidget(corpusLabel);
    infoColumn->addWidget(embeddingsLabel);
    minCountForm->addRow("Min word count", minCountSpin);
    minCountForm->addRow("Corpus storage", corpusStorageCombo);
    infoColumn->addLayout(minCountForm);
    infoColumn->setAlignment(Qt::AlignTop);

    infoRow->addLayout(infoColumn);
    infoRow->addWidget(configWidget);

    trainingBar->addWidget(toggleTrainingButton);
    trainingBar->addWidget(loadEmbeddingsButton);
    trainingBar->addWidget(saveEmbeddingsButton);
    trainingBar->addWidget(progressBar, 1);
    trainingBar->addWidget(etaLabel);
    trainingBar->addWidget(statusLabel);

    chartRow->addWidget(lossChart);
    chartRow->addWidget(speedChart);

    root->addLayout(dataButtonBar);
    root->addLayout(infoRow);
    root->addLayout(trainingBar);
    root->addLayout(chartRow, 1);
    setLayout(root);
}

void WordsDataTabWidget::setController(WordsController* newController)
{
    controller = newController;

    configWidget->setConfig(controller->getInfo().config);
    minCountSpin->setValue(static_cast<int>(controller->getInfo().minCount));
    corpusStorageCombo->setCurrentIndex(corpusStorageCombo->findData(
        static_cast<int>(controller->getInfo().corpusStorage)));

    connect(openDumpButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Open text dump", controller->getInfo().dumpPath, "Text files (*)");
        if (!path.isEmpty()) {
            controller->setDumpPath(path);
        }
    });

    connect(inspectButton, &QPushButton::clicked, [this] {
        controller->inspectDump();
    });

    connect(buildVocabularyButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getSaveFileName(
            this, "Save vocabulary", controller->getInfo().vocabularyPath, "Vocabulary (*.voc)");
        if (path.isEmpty()) {
            return;
        }
        controller->setVocabularyPath(path);
        controller->setMinCount(static_cast<std::size_t>(minCountSpin->value()));
        controller->buildVocabulary();
    });

    connect(loadVocabularyButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Load vocabulary", controller->getInfo().vocabularyPath, "Vocabulary (*.voc);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        controller->setVocabularyPath(path);
        controller->loadVocabulary();
    });

    connect(buildCorpusButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getSaveFileName(
            this, "Save corpus", controller->getInfo().corpusPath, "Corpus (*.cor)");
        if (path.isEmpty()) {
            return;
        }
        controller->setCorpusPath(path);
        controller->setCorpusStorage(selectedCorpusStorage());
        controller->buildCorpus();
    });

    connect(loadCorpusButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Load corpus", controller->getInfo().corpusPath, "Corpus (*.cor);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        controller->setCorpusPath(path);
        controller->setCorpusStorage(selectedCorpusStorage());
        controller->loadCorpus();
    });

    connect(toggleTrainingButton, &QPushButton::clicked, [this] {
        if (controller->getInfo().training) {
            controller->requestStop();
        } else {
            // Read at click time; see the rule in words_controller.h.
            controller->setConfig(configWidget->getConfig());
            controller->setMinCount(static_cast<std::size_t>(minCountSpin->value()));
            controller->startTraining();
        }
    });

    connect(loadEmbeddingsButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Load embeddings", controller->getInfo().embeddingsPath, "Embeddings (*.emb);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        controller->setEmbeddingsPath(path);
        controller->loadEmbeddings();
    });

    connect(saveEmbeddingsButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getSaveFileName(
            this, "Save embeddings", controller->getInfo().embeddingsPath, "Embeddings (*.emb)");
        if (path.isEmpty()) {
            return;
        }
        controller->setEmbeddingsPath(path);
        controller->saveEmbeddings();
    });
}

Words::CorpusStorage WordsDataTabWidget::selectedCorpusStorage() const
{
    return static_cast<Words::CorpusStorage>(corpusStorageCombo->currentData().toInt());
}

void WordsDataTabWidget::updateInfo(const WordsController::Info& info)
{
    dumpLabel->setText(info.dumpPath.isEmpty() ? "No text dump" : "Dump: " + info.dumpPath);
    vocabularyLabel->setText(info.hasVocabulary
        ? QString("Vocabulary: %1 words (kept %2 of %3 tokens)")
              .arg(info.vocabularySize).arg(info.keptTokens).arg(info.rawTokens)
        : QString("No vocabulary"));
    corpusLabel->setText(info.hasCorpus
        ? QString("Corpus: %1 tokens").arg(info.corpusTokens)
        : QString("No corpus"));
    embeddingsLabel->setText(info.hasEmbeddings
        ? QString("Embeddings: dim %1").arg(info.embeddingDim)
        : QString("No embeddings"));

    const bool idle = !info.busy;
    openDumpButton->setEnabled(idle);
    inspectButton->setEnabled(idle && !info.dumpPath.isEmpty());
    buildVocabularyButton->setEnabled(idle && !info.dumpPath.isEmpty());
    loadVocabularyButton->setEnabled(idle);
    buildCorpusButton->setEnabled(idle && info.hasVocabulary && !info.dumpPath.isEmpty());
    loadCorpusButton->setEnabled(idle && info.hasVocabulary);
    loadEmbeddingsButton->setEnabled(idle && info.hasVocabulary);
    saveEmbeddingsButton->setEnabled(idle && info.hasEmbeddings);
    configWidget->setDisabled(!idle);
    minCountSpin->setDisabled(!idle);
    corpusStorageCombo->setDisabled(!idle);

    toggleTrainingButton->setText(info.training ? "Stop training" : "Start training");
    toggleTrainingButton->setEnabled(
        info.training || (idle && info.hasVocabulary && info.hasCorpus));

    statusLabel->setText(info.busyLabel);
    if (info.busy && !info.training) {
        progressBar->setRange(0, 0);   // indeterminate
    } else if (!info.busy) {
        progressBar->setRange(0, 1000);
        progressBar->setValue(0);
        etaLabel->clear();
    } else {
        progressBar->setRange(0, 1000);   // training: fed by handleTrainProgress
    }
}

void WordsDataTabWidget::handleTrainProgress(const Words::TrainProgress& progress)
{
    lossChart->addPoint(progress.loss, "loss");
    speedChart->addPoint(progress.pairsPerSecond / 1e6, "pairs/s");
    progressBar->setValue(static_cast<int>(progress.progress * 1000));
    etaLabel->setText(QString("ETA: %1m %2s")
        .arg(static_cast<int>(progress.etaSeconds) / 60)
        .arg(static_cast<int>(progress.etaSeconds) % 60));
}
