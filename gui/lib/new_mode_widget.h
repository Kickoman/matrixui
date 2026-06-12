#pragma once


#include "gui/lib/mode_widget.h"
#include "gui/lib/mode_factory.h"


class NewModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit NewModeWidget(QWidget* parent = nullptr);

signals:
    void modeRequested(ModeType type);
};
