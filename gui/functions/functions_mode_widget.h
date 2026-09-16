#pragma once

#include "gui/functions/functions_controller.h"
#include "gui/functions/functions_points_model.h"
#include "gui/lib/mode_widget.h"

#include <memory>
#include <ostream>

class AdvancedTerminal;
class FunctionPlotWidget;
class FunctionsConfigWidget;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QWidget;


class FunctionsModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit FunctionsModeWidget(QWidget* parent = nullptr);
    ~FunctionsModeWidget() override;

    void setController(FunctionsController* value);
    std::ostream* getTerminalStream();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void updateInfo();
    void handleSnapshot(const FunctionsSnapshot& snapshot);
    void handlePointsChanged();
    void handleRowChanged(int row);
    void handleTableEdited(int row, int column);

private:
    void buildLayout();
    void connectController();
    void pushStateToController();
    void rebuildTableColumns();
    void refreshTable();
    void rebuildCurves();

    FunctionsController* controller = nullptr;
    FunctionsPointsModel model;
    std::unique_ptr<std::ostream> terminalStream;
    FunctionsSnapshot lastSnapshot;

    bool refreshingTable = false;

    FunctionsConfigWidget* configWidget = nullptr;
    QPushButton* loadConfigButton = nullptr;
    QPushButton* saveConfigButton = nullptr;
    QSpinBox* topCurvesSpin = nullptr;
    QPushButton* startButton = nullptr;
    QPushButton* resumeButton = nullptr;
    QPushButton* stopButton = nullptr;
    QLabel* statusLabel = nullptr;

    QWidget* plotPanel = nullptr;
    FunctionPlotWidget* plot = nullptr;
    QPushButton* fitViewButton = nullptr;

    QLineEdit* variablesEdit = nullptr;
    QTableWidget* pointsTable = nullptr;
    QPushButton* addPointButton = nullptr;
    QPushButton* deletePointButton = nullptr;
    QPushButton* clearPointsButton = nullptr;
    QPushButton* importCsvButton = nullptr;
    QPushButton* exportCsvButton = nullptr;

    QTableWidget* worldTable = nullptr;
    AdvancedTerminal* terminal = nullptr;
};
