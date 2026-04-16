#pragma once

#include <QMainWindow>

#include "gui/lib/mode_factory.h"


class QTabWidget;
class MainController;
class AddTabWidget;


class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    void setController(MainController* controller);

    void closeEvent(QCloseEvent* event) override;

signals:
    void newModeRequested(ModeType type);

public slots:
    void handleNewTabRequested();
    void handleCloseTabRequested(int index);

private:
    MainController* controller = nullptr;
    AddTabWidget* mainTabWidget = nullptr;
};
