#include "gui/main_window.h"
#include "gui/main_controller.h"

#include "gui_common/add_tab_widget.h"

#include "gui/lib/mode_container_widget.h"

#include <QVBoxLayout>
#include <QSplitter>
#include <QPushButton>
#include <QTabWidget>
#include <QTabBar>
#include <QLabel>


MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    mainTabWidget = new AddTabWidget(this);
    mainTabWidget->setTabsClosable(true);

    setCentralWidget(mainTabWidget);
    connect(mainTabWidget, &AddTabWidget::newTabRequested, this, &MainWindow::handleNewTabRequested);
    connect(mainTabWidget, &QTabWidget::tabCloseRequested, this, &MainWindow::handleCloseTabRequested);
    showMaximized();
}

MainWindow::~MainWindow() {}


void MainWindow::setController(MainController* controller) {
    this->controller = controller;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    QList<QWidget*> openTabs;
    openTabs.reserve(mainTabWidget->count());
    for (int i = 0; i < mainTabWidget->count(); ++i) {
        openTabs.append(mainTabWidget->widget(i));
    }
    for (auto* tab : openTabs) {
        if (const auto index = mainTabWidget->indexOf(tab); index != -1) {
            handleCloseTabRequested(index);
        }
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::handleNewTabRequested() {
    auto* container = new ModeContainerWidget();
    mainTabWidget->addTab(container, "New mode");
    mainTabWidget->setCurrentWidget(container);

    connect(container, &ModeContainerWidget::modeRequested, controller, &MainController::handleNewModeRequest);
    connect(container, &ModeContainerWidget::modeNameChanged, [this, container](const QString& newName){
        const auto idx = mainTabWidget->indexOf(container);
        if (idx != -1) {
            mainTabWidget->setTabText(idx, newName);
        }
    });
}

void MainWindow::handleCloseTabRequested(const int index) {
    auto* container = qobject_cast<ModeContainerWidget*>(mainTabWidget->widget(index));
    if (!container) {
        return;
    }
    mainTabWidget->tabBar()->setTabButton(index, QTabBar::ButtonPosition::RightSide, nullptr);
    connect(controller, &MainController::modeReadyToClose, [this, container](ModeContainerWidget* readyContainer){
        if (readyContainer != container) {
            return;
        }
        const auto index = mainTabWidget->indexOf(container);
        mainTabWidget->removeTab(index);
        container->deleteLater();
        if (mainTabWidget->count() == 0) {
            QMetaObject::invokeMethod(this, &MainWindow::handleNewTabRequested, Qt::QueuedConnection);
        }
    });
    controller->requestModeStop(container);
}
