#include "gui/functions/functions_mode_widget.h"

#include "core/functions/expected_csv.h"
#include "gui/functions/function_plot_widget.h"
#include "gui/functions/functions_config_widget.h"
#include "gui_common/advanced_terminal.h"

#include "core/lib/rpn_compile.h"

#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QVBoxLayout>

#include <fstream>
#include <limits>
#include <memory>


FunctionsModeWidget::FunctionsModeWidget(QWidget* parent)
    : ModeWidget("Functions", parent)
{
    buildLayout();

    connect(&model, &FunctionsPointsModel::changed, this, &FunctionsModeWidget::handlePointsChanged);
    connect(&model, &FunctionsPointsModel::rowChanged, this, &FunctionsModeWidget::handleRowChanged);
    connect(plot, &FunctionPlotWidget::pointAdded, &model, &FunctionsPointsModel::addPoint);
    connect(plot, &FunctionPlotWidget::pointMoved, &model, &FunctionsPointsModel::movePoint);
    connect(plot, &FunctionPlotWidget::pointDeleted, &model, &FunctionsPointsModel::removeRow);
    connect(fitViewButton, &QPushButton::clicked, plot, &FunctionPlotWidget::resetView);

    connect(pointsTable, &QTableWidget::cellChanged, this, &FunctionsModeWidget::handleTableEdited);
    connect(addPointButton, &QPushButton::clicked, &model, &FunctionsPointsModel::addRow);
    connect(deletePointButton, &QPushButton::clicked, [this] {
        model.removeRow(pointsTable->currentRow());
    });
    connect(clearPointsButton, &QPushButton::clicked, &model, &FunctionsPointsModel::clear);

    connect(variablesEdit, &QLineEdit::editingFinished, [this] {
        QStringList names;
        for (const auto& part : variablesEdit->text().split(',')) {
            const auto trimmed = part.trimmed();
            if (!trimmed.isEmpty()) {
                names.append(trimmed);
            }
        }
        const auto duplicated = QStringList(names).removeDuplicates() > 0;
        if (names.isEmpty() || duplicated) {
            *getTerminalStream() << "Variable names must be non-empty and unique." << std::endl;
            variablesEdit->setText(model.getVariables().join(", "));
            return;
        }
        model.setVariables(names);
    });

    rebuildTableColumns();
    refreshTable();
}

FunctionsModeWidget::~FunctionsModeWidget() {
    if (controller != nullptr) {
        pushStateToController();
        controller->saveSettings();
        controller->requestStop();
    }
}

void FunctionsModeWidget::buildLayout() {
    terminal = new AdvancedTerminal(this);
    terminal->setMaximumHeight(250);

    // --- left: configuration and run controls ---
    configWidget = new FunctionsConfigWidget(this);
    loadConfigButton = new QPushButton("Load config...", this);
    saveConfigButton = new QPushButton("Save config...", this);
    topCurvesSpin = new QSpinBox(this);
    topCurvesSpin->setRange(0, FunctionsController::kMaxCurves);
    topCurvesSpin->setValue(3);
    startButton = new QPushButton("Start", this);
    resumeButton = new QPushButton("Resume", this);
    stopButton = new QPushButton("Stop", this);
    statusLabel = new QLabel("Idle", this);
    statusLabel->setWordWrap(true);

    auto* configButtons = new QHBoxLayout();
    configButtons->addWidget(loadConfigButton);
    configButtons->addWidget(saveConfigButton);

    auto* curvesForm = new QFormLayout();
    curvesForm->addRow("Curves shown", topCurvesSpin);

    auto* runButtons = new QHBoxLayout();
    runButtons->addWidget(startButton);
    runButtons->addWidget(resumeButton);
    runButtons->addWidget(stopButton);

    auto* leftPanel = new QWidget(this);
    leftPanel->setMaximumWidth(470);
    auto* leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->addWidget(configWidget);
    leftLayout->addLayout(configButtons);
    leftLayout->addLayout(curvesForm);
    leftLayout->addLayout(runButtons);
    leftLayout->addWidget(statusLabel);
    leftLayout->addStretch(1);

    // --- centre: the interactive plot, single-variable only ---
    plot = new FunctionPlotWidget(this);
    fitViewButton = new QPushButton("Fit view", this);
    auto* hint = new QLabel(
        "click: add point · drag: move · right-click: delete · drag empty: pan · wheel: zoom", this);
    hint->setWordWrap(true);

    auto* plotButtons = new QHBoxLayout();
    plotButtons->addWidget(fitViewButton);
    plotButtons->addWidget(hint, 1);

    plotPanel = new QWidget(this);
    auto* plotLayout = new QVBoxLayout(plotPanel);
    plotLayout->setContentsMargins(0, 0, 0, 0);
    plotLayout->addWidget(plot, 1);
    plotLayout->addLayout(plotButtons);

    // --- points ---
    variablesEdit = new QLineEdit("x", this);
    variablesEdit->setToolTip("Comma-separated variable names; the plot appears with exactly one.");
    auto* variablesForm = new QFormLayout();
    variablesForm->addRow("Variables", variablesEdit);

    pointsTable = new QTableWidget(this);
    pointsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    addPointButton = new QPushButton("Add", this);
    deletePointButton = new QPushButton("Delete", this);
    clearPointsButton = new QPushButton("Clear", this);
    importCsvButton = new QPushButton("Import CSV...", this);
    exportCsvButton = new QPushButton("Export CSV...", this);

    auto* pointButtons = new QHBoxLayout();
    pointButtons->addWidget(addPointButton);
    pointButtons->addWidget(deletePointButton);
    pointButtons->addWidget(clearPointsButton);
    auto* csvButtons = new QHBoxLayout();
    csvButtons->addWidget(importCsvButton);
    csvButtons->addWidget(exportCsvButton);

    auto* pointsPanel = new QWidget(this);
    auto* pointsLayout = new QVBoxLayout(pointsPanel);
    pointsLayout->setContentsMargins(0, 0, 0, 0);
    pointsLayout->addLayout(variablesForm);
    pointsLayout->addWidget(pointsTable, 1);
    pointsLayout->addLayout(pointButtons);
    pointsLayout->addLayout(csvButtons);

    // --- right: the world ---
    worldTable = new QTableWidget(this);
    worldTable->setColumnCount(4);
    worldTable->setHorizontalHeaderLabels({"Expression", "Rank", "Copies", "Birth"});
    worldTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    worldTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    worldTable->setAlternatingRowColors(true);
    worldTable->horizontalHeader()->setStretchLastSection(false);
    worldTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int column = 1; column < worldTable->columnCount(); ++column) {
        worldTable->horizontalHeader()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
    }

    auto* worldPanel = new QWidget(this);
    auto* worldLayout = new QVBoxLayout(worldPanel);
    worldLayout->setContentsMargins(0, 0, 0, 0);
    worldLayout->addWidget(new QLabel("World", this));
    worldLayout->addWidget(worldTable, 1);

    auto* main = new QHBoxLayout();
    main->addWidget(leftPanel);
    main->addWidget(plotPanel, 3);
    main->addWidget(pointsPanel, 1);
    main->addWidget(worldPanel, 2);

    auto* root = new QVBoxLayout();
    root->addLayout(main, 1);
    root->addWidget(terminal);
    setLayout(root);
}

void FunctionsModeWidget::setController(FunctionsController* value) {
    controller = value;
    controller->loadSettings();

    const auto loadedPoints = controller->takeLoadedPoints();
    if (!loadedPoints.empty()) {
        model.setFromEntries(loadedPoints);
    }

    const auto info = controller->getInfo();
    configWidget->setConfig(info.config);
    topCurvesSpin->setValue(info.topCurves);
    variablesEdit->setText(model.getVariables().join(", "));
    plot->resetView();

    connectController();
    updateInfo();
}

void FunctionsModeWidget::connectController() {
    connect(controller, &FunctionsController::infoUpdated, this, &FunctionsModeWidget::updateInfo);
    connect(controller, &FunctionsController::snapshotReady,
            this, &FunctionsModeWidget::handleSnapshot);

    connect(configWidget, &FunctionsConfigWidget::edited,
            controller, &FunctionsController::markSessionDirty);
    connect(topCurvesSpin, &QSpinBox::valueChanged, [this](const int value) {
        controller->setTopCurves(value);
        rebuildCurves();
    });

    connect(startButton, &QPushButton::clicked, [this] {
        pushStateToController();
        plot->setCurves({});
        controller->start();
    });
    connect(resumeButton, &QPushButton::clicked, controller, &FunctionsController::resume);
    connect(stopButton, &QPushButton::clicked, controller, &FunctionsController::requestStop);

    connect(loadConfigButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Load configuration", controller->getInfo().configPath,
            "JSON config (*.json);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        controller->setConfigPath(path);
        if (controller->loadConfigFile(path)) {
            configWidget->setConfig(controller->getInfo().config);
        }
    });
    connect(saveConfigButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getSaveFileName(
            this, "Save configuration", controller->getInfo().configPath,
            "JSON config (*.json);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        controller->setConfig(configWidget->getConfig());
        controller->setConfigPath(path);
        controller->saveConfigFile(path);
    });

    connect(importCsvButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, "Import data points", controller->getInfo().dataPath,
            "CSV data (*.csv);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        try {
            std::ifstream file(path.toStdString());
            if (!file) {
                throw std::runtime_error("cannot open the file");
            }
            model.setFromEntries(Genetizer::ParseExpectedCsv(file));
            variablesEdit->setText(model.getVariables().join(", "));
            controller->setDataPath(path);
            plot->resetView();
        } catch (const std::exception& error) {
            *getTerminalStream() << "CSV import failed: " << error.what() << std::endl;
        }
    });
    connect(exportCsvButton, &QPushButton::clicked, [this] {
        const auto path = QFileDialog::getSaveFileName(
            this, "Export data points", controller->getInfo().dataPath,
            "CSV data (*.csv);;All files (*)");
        if (path.isEmpty()) {
            return;
        }
        try {
            std::ofstream file(path.toStdString());
            if (!file) {
                throw std::runtime_error("cannot write the file");
            }
            std::vector<std::string> names;
            for (const auto& name : model.getVariables()) {
                names.push_back(name.toStdString());
            }
            Genetizer::WriteExpectedCsv(file, names, model.toEntries());
            controller->setDataPath(path);
        } catch (const std::exception& error) {
            *getTerminalStream() << "CSV export failed: " << error.what() << std::endl;
        }
    });
}

void FunctionsModeWidget::pushStateToController() {
    controller->setConfig(configWidget->getConfig());
    controller->setTopCurves(topCurvesSpin->value());
    controller->setPoints(model.toEntries());
}

std::ostream* FunctionsModeWidget::getTerminalStream() {
    if (!terminalStream) {
        terminalStream = createTerminalOStream(terminal);
    }
    return terminalStream.get();
}

void FunctionsModeWidget::rebuildTableColumns() {
    refreshingTable = true;
    const auto& variables = model.getVariables();
    pointsTable->setColumnCount(variables.size() + 1);
    auto labels = variables;
    labels.append("expected");
    pointsTable->setHorizontalHeaderLabels(labels);
    pointsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    refreshingTable = false;
}

void FunctionsModeWidget::refreshTable() {
    refreshingTable = true;
    const auto columns = model.getVariables().size() + 1;
    pointsTable->setRowCount(model.getRowCount());
    for (int row = 0; row < model.getRowCount(); ++row) {
        for (int column = 0; column < columns; ++column) {
            const auto text = QString::number(model.getValue(row, column), 'g', 10);
            if (auto* item = pointsTable->item(row, column); item != nullptr) {
                item->setText(text);
            } else {
                pointsTable->setItem(row, column, new QTableWidgetItem(text));
            }
        }
    }
    refreshingTable = false;
}

void FunctionsModeWidget::handlePointsChanged() {
    if (pointsTable->columnCount() != model.getVariables().size() + 1) {
        rebuildTableColumns();
    }
    refreshTable();

    const auto single = model.getVariables().size() == 1;
    plotPanel->setVisible(single);
    plot->setPoints(model.toPlotPoints());

    if (controller != nullptr) {
        controller->markSessionDirty();
    }
}

void FunctionsModeWidget::handleRowChanged(const int row) {
    refreshingTable = true;
    const auto columns = model.getVariables().size() + 1;
    for (int column = 0; column < columns; ++column) {
        const auto text = QString::number(model.getValue(row, column), 'g', 10);
        if (auto* item = pointsTable->item(row, column); item != nullptr) {
            item->setText(text);
        } else {
            pointsTable->setItem(row, column, new QTableWidgetItem(text));
        }
    }
    refreshingTable = false;

    if (model.getVariables().size() == 1) {
        plot->setPoints(model.toPlotPoints());
    }
    if (controller != nullptr) {
        controller->markSessionDirty();
    }
}

void FunctionsModeWidget::handleTableEdited(const int row, const int column) {
    if (refreshingTable) {
        return;
    }
    auto* item = pointsTable->item(row, column);
    if (item == nullptr) {
        return;
    }
    bool ok = false;
    const auto value = item->text().toDouble(&ok);
    if (!ok) {
        refreshingTable = true;
        item->setText(QString::number(model.getValue(row, column), 'g', 10));
        refreshingTable = false;
        return;
    }
    model.setValue(row, column, value);
}

void FunctionsModeWidget::handleSnapshot(const FunctionsSnapshot& snapshot) {
    lastSnapshot = snapshot;

    worldTable->setRowCount(static_cast<int>(snapshot.rows.size()));
    for (std::size_t i = 0; i < snapshot.rows.size(); ++i) {
        const auto& row = snapshot.rows[i];
        const auto set = [this, i](const int column, const QString& text) {
            const auto index = static_cast<int>(i);
            if (auto* item = worldTable->item(index, column); item != nullptr) {
                item->setText(text);
            } else {
                worldTable->setItem(index, column, new QTableWidgetItem(text));
            }
        };
        set(0, row.expression);
        set(1, QString::number(row.rank, 'f', 4));
        set(2, QString::number(row.copies));
        set(3, QString::number(row.birth));
    }

    statusLabel->setText(QString("Running — epoch %1, unique %2 / %3")
                             .arg(snapshot.epoch)
                             .arg(snapshot.uniqueCount)
                             .arg(snapshot.totalCount));
    rebuildCurves();
}

void FunctionsModeWidget::rebuildCurves() {
    if (model.getVariables().size() != 1) {
        plot->setCurves({});
        return;
    }

    const auto wanted = std::min<std::size_t>(topCurvesSpin->value(), lastSnapshot.rows.size());
    const auto variable = model.getVariables().front().toStdString();

    std::vector<FunctionPlotWidget::Curve> curves;
    curves.reserve(wanted);
    for (std::size_t i = 0; i < wanted; ++i) {
        const auto& row = lastSnapshot.rows[i];
        auto program = std::make_shared<Matematyka::CompiledExpression<double>>();
        const auto status = program->compile(
            row.expressionCopy->getRpn(),
            [&variable](const std::string& name) -> std::uint32_t {
                return name == variable ? 1 : 0;
            });
        const bool runnable = status == Matematyka::CompiledExpression<double>::Status::Ok;
        auto stack = std::make_shared<std::vector<double>>(runnable ? program->getStackDepth() + 1 : 1);

        curves.push_back(FunctionPlotWidget::Curve{
            .label = QString("%1  (%2)").arg(row.expression).arg(row.rank, 0, 'f', 4),

            .evaluate = [program, stack, runnable](const double x) -> double {
                if (!runnable) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double values[] = {0.0, x};
                return program->eval(values, stack->data());
            },
        });
    }
    plot->setCurves(std::move(curves));
}

void FunctionsModeWidget::updateInfo() {
    if (controller == nullptr) {
        return;
    }
    const auto info = controller->getInfo();
    const auto idle = !info.running;

    configWidget->setEnabled(idle);
    loadConfigButton->setEnabled(idle);
    saveConfigButton->setEnabled(idle);
    variablesEdit->setEnabled(idle);
    pointsTable->setEnabled(idle);
    addPointButton->setEnabled(idle);
    deletePointButton->setEnabled(idle);
    clearPointsButton->setEnabled(idle);
    importCsvButton->setEnabled(idle);
    exportCsvButton->setEnabled(idle);

    startButton->setText(info.hasSession ? "Restart" : "Start");
    startButton->setEnabled(idle);
    resumeButton->setEnabled(idle && info.canResume);
    stopButton->setEnabled(info.running);

    plot->setEditable(idle);

    if (info.running) {
        return;
    }
    if (info.hasSession && info.canResume) {
        statusLabel->setText(QString("Stopped at epoch %1 — Resume continues, Restart reseeds")
                                 .arg(info.epoch));
    } else if (info.hasSession) {
        statusLabel->setText(QString("Edited after epoch %1 — Restart only").arg(info.epoch));
    } else {
        statusLabel->setText("Idle");
    }
}

void FunctionsModeWidget::closeEvent(QCloseEvent* event) {
    if (controller != nullptr) {
        pushStateToController();
        controller->saveSettings();
        controller->requestStop();
    }
    QWidget::closeEvent(event);
}
