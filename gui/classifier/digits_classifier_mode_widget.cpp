#include "gui/classifier/digits_classifier_mode_widget.h"

#include "gui_common/advanced_terminal.h"
#include "gui_common/time_chart.h"
#include "gui_common/digit_chart.h"

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

    toggleLearningButton = new QPushButton("Start learning", this);
    openNetworkButton = new QPushButton("Open network", this);
    openTrainingDatasetButton = new QPushButton("Open training dataset", this);
    openTestingDatasetButton = new QPushButton("Open testing dataset", this);
    currentNetworkLabel = new QLabel("No network", this);
    currentTestingDatasetLabel = new QLabel("No testing dataset", this);
    currentTrainingDatasetLabel = new QLabel("No training dataset", this);
    learningConfigWidget = new LearningConfigWidget(this);
    learningConfigWidget->setConfig({});

    auto* vLayout = new QVBoxLayout();
    auto* hLayout = new QHBoxLayout();
    auto* buttonLayout = new QHBoxLayout();
    auto* infoLayout = new QHBoxLayout();
    auto* networkInfoLayout = new QVBoxLayout();
    auto* configInfoLayout = new QVBoxLayout();
    buttonLayout->addWidget(toggleLearningButton);
    buttonLayout->addWidget(openNetworkButton);
    buttonLayout->addWidget(openTrainingDatasetButton);
    buttonLayout->addWidget(openTestingDatasetButton);

    infoLayout->addLayout(networkInfoLayout);
    infoLayout->addWidget(learningConfigWidget);
    networkInfoLayout->addWidget(currentNetworkLabel);
    networkInfoLayout->addWidget(currentTestingDatasetLabel);
    networkInfoLayout->addWidget(currentTrainingDatasetLabel);
    networkInfoLayout->setAlignment(Qt::AlignTop);

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
        } else {
            controller->run(learningConfigWidget->getConfig());
        }
    });
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

    for (unsigned i = 0; i < 10; ++i) {
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
}

void DigitsClassifierModeWidget::handleOpenNetworkClicked() {
    NetworkCreateDialog dialog(this);
    dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().networkName));
    if (dialog.exec() == QDialog::Accepted) {
        const QString name = dialog.getNetworkName();
        const auto config = dialog.getConfiguration();
        controller->loadNetwork(name, config);
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
