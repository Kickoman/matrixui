#pragma once

#include <QWidget>
#include "gui/lib/mode_factory.h"

class ModeWidget;
class NewModeWidget;
class QStackedLayout;

class ModeContainerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ModeContainerWidget(QWidget* parent = nullptr);

    QString getModeName() const;

signals:
    void modeRequested(ModeContainerWidget* self, ModeType mode);
    void modeNameChanged(const QString& newModeName);

public slots:
    void setModeWidget(ModeWidget* modeWidget);

private:
    QStackedLayout* mainLayout = nullptr;
};
