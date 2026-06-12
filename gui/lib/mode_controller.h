#pragma once

#include <QObject>


class ModeController : public QObject
{
    Q_OBJECT
public:
    explicit ModeController(QObject* parent = nullptr)
        : QObject(parent) {}

    virtual void waitUntilFinished() {}

public slots:
    virtual void requestStop() {}
};
