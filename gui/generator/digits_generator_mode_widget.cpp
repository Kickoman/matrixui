#include "gui/generator/digits_generator_mode_widget.h"

#include "gui_common/advanced_terminal.h"
#include "gui_common/time_chart.h"

#include "gui/generator/digits_generator_controller.h"
#include "gui/generator/gan_config_widget.h"

#include "gui/lib/theme.h"
#include "gui/lib/network_create_dialog.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QSpinBox>
#include <QChart>


DigitsGeneratorModeWidget::DigitsGeneratorModeWidget(QWidget* parent)
    : ModeWidget("GAN Generator", parent)
{
    terminal = new AdvancedTerminal(this);
    terminal->setMaximumHeight(250);

    lossChart = new TimeChart(this);
    lossChart->setTitle("Training scores");
    AppTheme::ApplyTheme(lossChart->getChart());

    toggleTrainingButton = new QPushButton("Start training", this);
    loadClassifierButton = new QPushButton("Load classifier", this);
    loadDatasetButton = new QPushButton("Load dataset", this);
    loadGeneratorButton = new QPushButton("Set generator file", this);
    loadDiscriminatorButton = new QPushButton("Set discriminator file", this);

    classifierLabel = new QLabel("No classifier", this);
    datasetLabel = new QLabel("No dataset", this);
    generatorLabel = new QLabel("Generator: generator.wgt", this);
    discriminatorLabel = new QLabel("Discriminator: discriminator.wgt", this);

    imageWidth = new QSpinBox(this);
    imageHeight = new QSpinBox(this);
    imageWidth->setRange(1, 1000);
    imageHeight->setRange(1, 1000);
    imageWidth->setValue(28);
    imageHeight->setValue(28);

    ganConfigWidget = new GanConfigWidget(this);
    previewLabel = new QLabel(this);
    previewLabel->setFixedSize(140, 140);
    previewLabel->setAlignment(Qt::AlignCenter);
    previewLabel->setStyleSheet("border: 1px solid gray;");

    digitSelector = new QComboBox(this);
    generateButton = new QPushButton("Generate", this);

    auto* root = new QVBoxLayout();

    auto* buttonBar = new QHBoxLayout();
    buttonBar->addWidget(toggleTrainingButton);
    buttonBar->addWidget(loadClassifierButton);
    buttonBar->addWidget(loadDatasetButton);
    buttonBar->addWidget(loadGeneratorButton);
    buttonBar->addWidget(loadDiscriminatorButton);
    root->addLayout(buttonBar);

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

    auto* imagePropertiesLayout = new QFormLayout();
    imagePropertiesLayout->addRow("Image width", imageWidth);
    imagePropertiesLayout->addRow("Image height", imageHeight);
    infoLeft->addLayout(imagePropertiesLayout);

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

    connect(loadClassifierButton, &QPushButton::clicked, [this] {
        const QString path = QFileDialog::getOpenFileName(this, "Load classifier weights", {}, "Network weights (*.wgt)");
        if (!path.isEmpty()) {
            controller->setClassifierPath(path);
        }
    });

    connect(loadDatasetButton, &QPushButton::clicked, [this] {
        QFileDialog dlg(this);
        dlg.setFileMode(QFileDialog::Directory);
        if (!dlg.exec()) {
            return;
        }

        const auto selected = dlg.selectedFiles();
        if (selected.size() != 1) {
            return;
        }
        controller->setDatasetPath(selected.front());
    });

    connect(loadGeneratorButton, &QPushButton::clicked, [this] {
        const auto ganConfig = ganConfigWidget->getConfig();
        const std::size_t numClasses = controller->getInfo().numClasses;
        const std::size_t inputSize = ganConfig.latentDim + (numClasses > 0 ? numClasses : 10);
        NetworkCreateDialog dialog(this);
        dialog.setWindowTitle("Open generator network");
        dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().generatorPath));
        dialog.setDefaultLayersText(QString("%1, 256, 512, 784").arg(inputSize));
        dialog.setDefaultActivations(Neural::ActivationType::LeakyReLU, Neural::ActivationType::Sigmoid);
        if (dialog.exec() == QDialog::Accepted) {
            controller->setGeneratorPath(dialog.getNetworkName(), dialog.getConfiguration());
        }
    });

    connect(loadDiscriminatorButton, &QPushButton::clicked, [this] {
        NetworkCreateDialog dialog(this);
        dialog.setWindowTitle("Open discriminator network");
        dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().discriminatorPath));
        dialog.setDefaultLayersText("784, 512, 256, 1");
        dialog.setDefaultActivations(Neural::ActivationType::LeakyReLU, Neural::ActivationType::Sigmoid);
        if (dialog.exec() == QDialog::Accepted) {
            controller->setDiscriminatorPath(dialog.getNetworkName(), dialog.getConfiguration());
        }
    });

    connect(generateButton, &QPushButton::clicked, this, &DigitsGeneratorModeWidget::handleGenerateClicked);
}

DigitsGeneratorModeWidget::~DigitsGeneratorModeWidget() {
    controller->saveSettings();
    controller->requestStop();
}

std::ostream* DigitsGeneratorModeWidget::getTerminalStream() {
    if (!terminalStream) {
        terminalStream = createTerminalOStream(terminal);
    }
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

    const auto updateImageProperties = [this](){
        this->controller->setImageHeight(imageHeight->value());
        this->controller->setImageWidth(imageWidth->value());
    };
    connect(imageWidth, &QSpinBox::valueChanged, updateImageProperties);
    connect(imageHeight, &QSpinBox::valueChanged, updateImageProperties);


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

    toggleTrainingButton->setText(info.running ? "Stop training" : "Start training");

    const bool idle = !info.running;
    ganConfigWidget->setDisabled(!idle);
    loadClassifierButton->setDisabled(!idle);
    loadDatasetButton->setDisabled(!idle);
    loadGeneratorButton->setDisabled(!idle);
    loadDiscriminatorButton->setDisabled(!idle);
    generateButton->setEnabled(idle);

    imageWidth->setValue(info.imageWidth);
    imageHeight->setValue(info.imageHeight);

    if (info.numClasses > 0 && static_cast<std::size_t>(digitSelector->count()) != info.numClasses) {
        const int prev = digitSelector->currentIndex();
        digitSelector->clear();
        for (std::size_t i = 0; i < info.numClasses; ++i) {
            digitSelector->addItem(QString::number(i));
        }
        digitSelector->setCurrentIndex(std::min(prev, digitSelector->count() - 1));
    }
}

void DigitsGeneratorModeWidget::handleEpochCompleted(std::size_t /*epoch*/, double dScore, double gScore, double emaReal, double emaGen) {
    lossChart->addPoint(dScore * 100, "D(real)");
    lossChart->addPoint(gScore * 100, "D(G(z))");
    lossChart->addPoint(emaReal * 100, "EMA D(real)");
    lossChart->addPoint(emaGen * 100, "EMA D(G(z))");
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
