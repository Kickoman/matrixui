#pragma once

#include <QTabBar>

class QToolButton;

class AddTabBar : public QTabBar
{
    Q_OBJECT
public:
    explicit AddTabBar(QWidget* parent = nullptr);

signals:
    void tabLayoutChanged();

protected:
    void tabLayoutChange() override;
};
