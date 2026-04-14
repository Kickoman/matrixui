#pragma once


#include "mode_widget.h"
#include "mode_factory.h"


class NewModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit NewModeWidget(QWidget* parent = nullptr);

signals:
    void modeRequested(ModeType type);
};
