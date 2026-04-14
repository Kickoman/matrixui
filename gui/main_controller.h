#pragma once

#include <QObject>

#include "mode_factory.h"

class QWidget;
class ModeContainerWidget;

class MainController : public QObject
{
    Q_OBJECT
public:
    explicit MainController(QObject* parent = nullptr);

    void requestModeStop(ModeContainerWidget* container);

signals:
    void modeReadyToClose(ModeContainerWidget* container);

public slots:
    void handleNewModeRequest(ModeContainerWidget* container, ModeType type);


private:
    std::unordered_map<QWidget*, Mode> modes;
};
