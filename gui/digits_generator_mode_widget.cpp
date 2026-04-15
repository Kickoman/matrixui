#include "digits_generator_mode_widget.h"
#include "digits_generator_controller.h"
#include "advanced_terminal.h"
#include "time_chart.h"
#include "gan_config_widget.h"
#include "network_create_dialog.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>


DigitsGeneratorModeWidget::DigitsGeneratorModeWidget(QWidget* parent)
    : ModeWidget("GAN Generator", parent)
{
    terminal = new AdvancedTerminal(this);
    terminal->setMaximumHeight(250);

    lossChart = new TimeChart(this);
    lossChart->setTitle("Training scores");

    toggleTrainingButton   = new QPushButton("Start training", this);
    loadClassifierButton   = new QPushButton("Load classifier", this);
    loadDatasetButton      = new QPushButton("Load dataset", this);
    loadGeneratorButton    = new QPushButton("Set generator file", this);
    loadDiscriminatorButton = new QPushButton("Set discriminator file", this);

    classifierLabel    = new QLabel("No classifier", this);
    datasetLabel       = new QLabel("No dataset", this);
    generatorLabel     = new QLabel("Generator: generator.wgt", this);
    discriminatorLabel = new QLabel("Discriminator: discriminator.wgt", this);

    ganConfigWidget = new GanConfigWidget(this);

    previewLabel = new QLabel(this);
    previewLabel->setFixedSize(140, 140);
    previewLabel->setAlignment(Qt::AlignCenter);
    previewLabel->setStyleSheet("border: 1px solid gray;");

    digitSelector = new QComboBox(this);
    for (int i = 0; i < 10; ++i)
        digitSelector->addItem(QString::number(i));

    generateButton = new QPushButton("Generate", this);

    // --- Layout ---
    auto* root = new QVBoxLayout();

    // Button bar
    auto* buttonBar = new QHBoxLayout();
    buttonBar->addWidget(toggleTrainingButton);
    buttonBar->addWidget(loadClassifierButton);
    buttonBar->addWidget(loadDatasetButton);
    buttonBar->addWidget(loadGeneratorButton);
    buttonBar->addWidget(loadDiscriminatorButton);
    root->addLayout(buttonBar);

    // Info labels
    auto* infoRow = new QHBoxLayout();
    auto* infoLeft = new QVBoxLayout();
    infoLeft->addWidget(classifierLabel);
    infoLeft->addWidget(datasetLabel);
    infoLeft->addWidget(generatorLabel);
    infoLeft->addWidget(discriminatorLabel);
    infoLeft->setAlignment(Qt::AlignTop);
    infoRow->addLayout(infoLeft);
    infoRow->addWidget(ganConfigWidget);
    root->addLayout(infoRow);

    // Chart + preview
    auto* midRow = new QHBoxLayout();
    midRow->addWidget(lossChart, 1);

    auto* previewPanel = new QVBoxLayout();
    previewPanel->addWidget(previewLabel, 0, Qt::AlignHCenter);
    previewPanel->addWidget(digitSelector);
    previewPanel->addWidget(generateButton);
    previewPanel->addStretch();
    midRow->addLayout(previewPanel);
    root->addLayout(midRow, 1);

    root->addWidget(terminal);

    setLayout(root);

    // Connections (controller not yet set, connected in setController)
    connect(loadClassifierButton, &QPushButton::clicked, [this] {
        const QString path = QFileDialog::getOpenFileName(this, "Load classifier weights", {}, "Network weights (*.wgt)");
        if (path.isEmpty()) return;
        if (!controller->loadClassifier(path))
            QMessageBox::critical(this, "Load failed", "Could not load classifier network from:\n" + path);
    });

    connect(loadDatasetButton, &QPushButton::clicked, [this] {
        QFileDialog dlg(this);
        dlg.setFileMode(QFileDialog::Directory);
        if (!dlg.exec()) return;
        const auto selected = dlg.selectedFiles();
        if (selected.size() != 1) return;
        if (!controller->loadDataset(selected.front()))
            QMessageBox::critical(this, "Invalid dataset",
                "The dataset directory must contain subdirectories 0–9 with PNG images.");
    });

    connect(loadGeneratorButton, &QPushButton::clicked, [this] {
        const auto ganConfig = ganConfigWidget->getConfig();
        const std::size_t inputSize = ganConfig.latentDim + ganConfig.numClasses;
        NetworkCreateDialog dialog(this);
        dialog.setWindowTitle("Open generator network");
        dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().generatorPath));
        dialog.setDefaultLayersText(QString("%1, 256, 512, 784").arg(inputSize));
        dialog.setDefaultActivations(Neural::ActivationType::LeakyReLU, Neural::ActivationType::Sigmoid);
        if (dialog.exec() == QDialog::Accepted)
            controller->loadGenerator(dialog.getNetworkName(), dialog.getConfiguration());
    });

    connect(loadDiscriminatorButton, &QPushButton::clicked, [this] {
        NetworkCreateDialog dialog(this);
        dialog.setWindowTitle("Open discriminator network");
        dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().discriminatorPath));
        dialog.setDefaultLayersText("784, 512, 256, 1");
        dialog.setDefaultActivations(Neural::ActivationType::LeakyReLU, Neural::ActivationType::Sigmoid);
        if (dialog.exec() == QDialog::Accepted)
            controller->loadDiscriminator(dialog.getNetworkName(), dialog.getConfiguration());
    });

    connect(generateButton, &QPushButton::clicked, this, &DigitsGeneratorModeWidget::handleGenerateClicked);
}

DigitsGeneratorModeWidget::~DigitsGeneratorModeWidget() {
    controller->saveSettings();
    controller->requestStop();
}

std::ostream* DigitsGeneratorModeWidget::getTerminalStream() {
    if (!terminalStream)
        terminalStream = createTerminalOStream(terminal);
    return terminalStream.get();
}

void DigitsGeneratorModeWidget::setController(DigitsGeneratorController* ctrl) {
    controller = ctrl;
    controller->loadSettings();

    connect(controller, &DigitsGeneratorController::infoUpdated,
            this, &DigitsGeneratorModeWidget::updateInfo);
    connect(controller, &DigitsGeneratorController::epochCompleted,
            this, &DigitsGeneratorModeWidget::handleEpochCompleted);

    connect(toggleTrainingButton, &QPushButton::clicked, [this] {
        if (controller->getInfo().running) {
            controller->requestStop();
        } else {
            controller->run(ganConfigWidget->getConfig());
        }
    });

    updateInfo();
}

void DigitsGeneratorModeWidget::closeEvent(QCloseEvent* event) {
    controller->saveSettings();
    controller->requestStop();
    QWidget::closeEvent(event);
}

void DigitsGeneratorModeWidget::updateInfo() {
    if (!controller) return;
    const auto info = controller->getInfo();

    classifierLabel->setText(info.classifierPath.empty()
        ? "No classifier"
        : "Classifier: " + QString::fromStdString(info.classifierPath));
    datasetLabel->setText(info.datasetPath.empty()
        ? "No dataset"
        : "Dataset: " + QString::fromStdString(info.datasetPath));
    generatorLabel->setText("Generator: " + QString::fromStdString(info.generatorPath));
    discriminatorLabel->setText("Discriminator: " + QString::fromStdString(info.discriminatorPath));

    toggleTrainingButton->setEnabled(info.canRun);
    toggleTrainingButton->setText(info.running ? "Stop training" : "Start training");

    const bool idle = !info.running;
    ganConfigWidget->setDisabled(!idle);
    loadClassifierButton->setDisabled(!idle);
    loadDatasetButton->setDisabled(!idle);
    loadGeneratorButton->setDisabled(!idle);
    loadDiscriminatorButton->setDisabled(!idle);
    generateButton->setEnabled(idle);
}

void DigitsGeneratorModeWidget::handleEpochCompleted(std::size_t /*epoch*/, double dScore, double gScore) {
    lossChart->addPoint(dScore * 100, "D(real)");
    lossChart->addPoint(gScore * 100, "D(G(z))");
}

void DigitsGeneratorModeWidget::handleGenerateClicked() {
    const std::size_t label = static_cast<std::size_t>(digitSelector->currentIndex());
    const QImage img = controller->generateSample(label);
    if (img.isNull()) {
        previewLabel->setText("No generator");
        return;
    }
    previewLabel->setPixmap(
        QPixmap::fromImage(img).scaled(previewLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation)
    );
}
