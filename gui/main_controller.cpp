#include "main_controller.h"
#include "mode_container_widget.h"
#include "mode_widget.h"
#include "mode_controller.h"

#include <QPointer>


MainController::MainController(QObject* parent)
    : QObject(parent)
{}

void MainController::handleNewModeRequest(ModeContainerWidget* container, const ModeType type) {
    QPointer<ModeContainerWidget> guard(container);

    auto mode = CreateMode(type);

    if (!guard) {
        mode.controller->deleteLater();
        mode.view->deleteLater();
        return;
    }
    mode.controller->setParent(this);
    guard->setModeWidget(mode.view);
    modes[guard] = mode;
}

void MainController::requestModeStop(ModeContainerWidget* container) {
    auto modeIterator = modes.find(container);
    if (modeIterator == modes.end()) {
        emit modeReadyToClose(container);
        return;
    }

    auto& mode = modeIterator->second;
    mode.controller->requestStop();
    mode.controller->waitUntilFinished();
    mode.view->deleteLater();
    mode.controller->deleteLater();
    modes.erase(modeIterator);
    emit modeReadyToClose(container);
}
