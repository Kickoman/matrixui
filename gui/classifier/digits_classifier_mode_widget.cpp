#include "gui/classifier/digits_classifier_mode_widget.h"

#include "gui_common/advanced_terminal.h"
#include "gui_common/time_chart.h"
#include "gui_common/digit_chart.h"

#include "gui/lib/theme.h"
#include "gui/lib/network_create_dialog.h"
#include "gui/classifier/learning_config_widget.h"
#include "gui/classifier/digits_classifier_controller.h"

#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QtCharts/QtCharts>
#include <QTimer>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <stdexcept>


DigitsClassifierModeWidget::DigitsClassifierModeWidget(QWidget* parent)
    : ModeWidget("Digits Classifier", parent)
{
    terminal = new AdvancedTerminal(this);
    terminal->setMaximumHeight(300);

    chart = new TimeChart(this);
    digitChart = new DigitChart(this);
    chart->setTitle("Test passing rate");
    digitChart->setTitle("Tests passing per digit");

    AppTheme::ApplyTheme(chart->getChart());
    AppTheme::ApplyTheme(digitChart->getChart());

    toggleLearningButton = new QPushButton("Start learning", this);
    openNetworkButton = new QPushButton("Open network", this);
    openTrainingDatasetButton = new QPushButton("Open training dataset", this);
    openTestingDatasetButton = new QPushButton("Open testing dataset", this);
    currentNetworkLabel = new QLabel("No network", this);
    currentTestingDatasetLabel = new QLabel("No testing dataset", this);
    currentTrainingDatasetLabel = new QLabel("No training dataset", this);
    imageWidth = new QSpinBox(this);
    imageHeight = new QSpinBox(this);
    imageWidth->setMinimum(1);
    imageWidth->setMaximum(1000);
    imageHeight->setMinimum(1);
    imageWidth->setMaximum(1000);
    imageWidth->setValue(28);
    imageHeight->setValue(28);
    learningConfigWidget = new LearningConfigWidget(this);
    learningConfigWidget->setConfig({});

    auto* vLayout = new QVBoxLayout();
    auto* hLayout = new QHBoxLayout();
    auto* buttonLayout = new QHBoxLayout();
    auto* infoLayout = new QHBoxLayout();
    auto* networkInfoLayout = new QVBoxLayout();
    auto* imagePropertiesLayout = new QFormLayout();
    buttonLayout->addWidget(toggleLearningButton);
    buttonLayout->addWidget(openNetworkButton);
    buttonLayout->addWidget(openTrainingDatasetButton);
    buttonLayout->addWidget(openTestingDatasetButton);

    infoLayout->addLayout(networkInfoLayout);
    infoLayout->addWidget(learningConfigWidget);

    networkInfoLayout->addWidget(currentNetworkLabel);
    networkInfoLayout->addWidget(currentTestingDatasetLabel);
    networkInfoLayout->addWidget(currentTrainingDatasetLabel);
    networkInfoLayout->addLayout(imagePropertiesLayout);
    networkInfoLayout->setAlignment(Qt::AlignTop);

    imagePropertiesLayout->addRow("Image width", imageWidth);
    imagePropertiesLayout->addRow("Image height", imageHeight);

    vLayout->addLayout(buttonLayout);
    vLayout->addLayout(infoLayout);
    vLayout->addLayout(hLayout);
    vLayout->addWidget(terminal);
    hLayout->addWidget(chart);
    hLayout->addWidget(digitChart);

    setLayout(vLayout);

    connect(openNetworkButton, &QPushButton::clicked, this, &DigitsClassifierModeWidget::handleOpenNetworkClicked);
    connect(openTrainingDatasetButton, &QPushButton::clicked, this, &DigitsClassifierModeWidget::handleOpenDatasetClicked);
    connect(openTestingDatasetButton, &QPushButton::clicked, this, &DigitsClassifierModeWidget::handleOpenDatasetClicked);
}

DigitsClassifierModeWidget::~DigitsClassifierModeWidget()
{
    controller->saveSettings();
    controller->requestStop();
}

std::ostream* DigitsClassifierModeWidget::getTerminalStream() {
    if (!terminalStream) {
        terminalStream = createTerminalOStream(terminal);
    }
    return terminalStream.get();
}

void DigitsClassifierModeWidget::setController(DigitsClassifierController* controller) {
    this->controller = controller;
    controller->loadSettings();
    connect(controller, &DigitsClassifierController::updatedStatistics, this, &DigitsClassifierModeWidget::handleStatistics);
    connect(controller, &DigitsClassifierController::infoUpdated, this, &DigitsClassifierModeWidget::updateInfo);
    connect(toggleLearningButton, &QPushButton::clicked, [this, controller]{
        if (controller->getInfo().running) {
            controller->requestStop();
            return;
        }
        try {
            controller->run(learningConfigWidget->getConfig());
        } catch (const std::exception& e) {
            QMessageBox::critical(this, tr("Can't start training"), QString::fromUtf8(e.what()));
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

void DigitsClassifierModeWidget::closeEvent(QCloseEvent* event) {
    controller->saveSettings();
    controller->requestStop();
    QWidget::closeEvent(event);
}

void DigitsClassifierModeWidget::handleStatistics(const Neural::Classifier::TestResult& result) {
    const auto& total = result.getTotal();
    const double rate = total.totalTests > 0 ? 100.0 * total.passedTests / total.totalTests : 0;
    chart->addPoint(rate);

    assert(result.size() == digitChart->getCount());
    for (unsigned i = 0; i < result.size(); ++i) {
        const auto& res = result.stats[i];
        const double rate = res.totalTests > 0 ? 100.0 * res.passedTests / res.totalTests : 0;
        digitChart->setValue(i, rate);
    }
}

void DigitsClassifierModeWidget::updateInfo() {
    if (!controller) {
        return;
    }
    const auto& info = controller->getInfo();
    currentNetworkLabel->setText(info.networkName.size() ? ("Network: " + QString::fromStdString(info.networkName)) : QString("No network"));
    currentTestingDatasetLabel->setText(info.pathToTestingDataset.size() ? ("Testing dataset: " + QString::fromStdString(info.pathToTestingDataset)) : QString("No dataset"));
    currentTrainingDatasetLabel->setText(info.pathToTrainingDataset.size() ? ("Training dataset: " + QString::fromStdString(info.pathToTrainingDataset)) : QString("No dataset"));
    toggleLearningButton->setEnabled(info.initialized);
    toggleLearningButton->setText(
        info.running ? "Stop learning" : "Start learning"
    );
    learningConfigWidget->setDisabled(info.running);
    imageWidth->setValue(info.imageWidth);
    imageHeight->setValue(info.imageHeight);

    digitChart->setCount(info.layersConfiguration.back());
}

void DigitsClassifierModeWidget::handleOpenNetworkClicked() {
    NetworkCreateDialog dialog(this);
    dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().networkName));
    dialog.setDefaultLayersText(controller->getInfo().layersConfiguration);
    if (dialog.exec() == QDialog::Accepted) {
        const QString name = dialog.getNetworkName();
        const auto config = dialog.getConfiguration();
        try {
            controller->loadNetwork(name, config);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, tr("Open network"),
                                 tr("Failed to load network:\n%1\n\n%2")
                                     .arg(name, QString::fromUtf8(e.what())));
        }
    }
}

void DigitsClassifierModeWidget::handleOpenDatasetClicked() {
    QFileDialog dialog(this);
    dialog.setModal(true);
    dialog.setFileMode(QFileDialog::Directory);
    if (dialog.exec()) {
        const auto selected = dialog.selectedFiles();
        if (selected.size() != 1) {
            throw std::runtime_error("A single directory should be selected");
        }
        bool set = false;
        if (QObject::sender() == openTestingDatasetButton) {
            set = controller->setTestingDataset(selected.front());
        } else if (QObject::sender() == openTrainingDatasetButton) {
            set = controller->setTrainingDataset(selected.front());
        }
        if (!set) {
            QMessageBox::critical(
                this,
                "Incorrect dataset format!",
                "The dataset directory should contain 10 directories per each digit named with the digit"
            );
        }
    }
}
