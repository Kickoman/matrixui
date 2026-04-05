#include "main_window.h"
#include "advanced_terminal.h"
#include "digits_runner.h"
#include "matrix.h"
#include "time_chart.h"
#include "digit_chart.h"

#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QtCharts/QtCharts>
#include <QTimer>
#include <QRandomGenerator>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QSpacerItem>
#include <qlayoutitem.h>
#include <qnamespace.h>
#include <stdexcept>
#include "network_create_dialog.h"
#include "learning_config_widget.h"


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
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

    auto* mainWidget = new QWidget(this);
    mainWidget->setLayout(vLayout);

    setCentralWidget(mainWidget);
    showMaximized();

    connect(openNetworkButton, &QPushButton::clicked, this, &MainWindow::handleOpenNetworkClicked);
    connect(openTrainingDatasetButton, &QPushButton::clicked, this, &MainWindow::handleOpenDatasetClicked);
    connect(openTestingDatasetButton, &QPushButton::clicked, this, &MainWindow::handleOpenDatasetClicked);
}

MainWindow::~MainWindow()
{
    controller->requestStop();
}

AdvancedTerminal* MainWindow::getTerminalWidget() const {
    return terminal;
}

void MainWindow::setLogger(std::ostream* stream) {
    this->stream = stream;
}

void MainWindow::setController(DigitsRecognizerController* controller) {
    this->controller = controller;
    connect(controller, &DigitsRecognizerController::updatedStatistics, this, &MainWindow::handleStatistics);
    connect(controller, &DigitsRecognizerController::infoUpdated, this, &MainWindow::updateInfo);
    connect(toggleLearningButton, &QPushButton::clicked, [this, controller]{
        if (controller->getInfo().running) {
            controller->requestStop();
        } else {
            controller->run(learningConfigWidget->getConfig());
        }
    });
    updateInfo();
}

void MainWindow::closeEvent(QCloseEvent* event) {
    controller->requestStop();
    QSettings settings;
    settings.setValue("last_training_dataset_path", QString::fromStdString(controller->getInfo().pathToTrainingDataset));
    settings.setValue("last_testing_dataset_path", QString::fromStdString(controller->getInfo().pathToTestingDataset));
    settings.setValue("last_network_name", QString::fromStdString(controller->getInfo().networkName));
    QMainWindow::closeEvent(event);
}

void MainWindow::handleTimer()
{
    const double value = QRandomGenerator::global()->bounded(100.0);
    logger() << "Appending " << value << std::endl;
    chart->addPoint(value);

    for (unsigned i = 0; i < 10; ++i) {
        const double val = QRandomGenerator::global()->bounded(5.0);
        digitChart->setValue(i, val);
        logger() << "Set " << val << " for " << i << std::endl;
    }
}

void MainWindow::handleStatistics(const Neural::TestResult& result) {
    const auto& total = result.getTotal();
    const double rate = total.totalTests > 0 ? 100.0 * total.passedTests / total.totalTests : 0;
    chart->addPoint(rate);

    for (unsigned i = 0; i < 10; ++i) {
        const auto& res = result.stats[i];
        const double rate = res.totalTests > 0 ? 100.0 * res.passedTests / res.totalTests : 0;
        digitChart->setValue(i, rate);
    }
}

void MainWindow::updateInfo() {
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

void MainWindow::handleOpenNetworkClicked() {
    NetworkCreateDialog dialog(this);
    dialog.setCurrentNetworkPath(QString::fromStdString(controller->getInfo().networkName));
    if (dialog.exec() == QDialog::Accepted) {
        const QString name = dialog.getNetworkName();
        const auto sizes = dialog.getLayerSizes();
        controller->loadNetwork(name, sizes);
    }
}

void MainWindow::handleOpenDatasetClicked() {
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

std::ostream& MainWindow::logger() {
    return *stream;
}
