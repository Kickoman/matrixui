#pragma once

#include <QTabWidget>

class QToolButton;

class AddTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    explicit AddTabWidget(QWidget* parent = nullptr);

signals:
    void newTabRequested();

protected:
    void showEvent(QShowEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private:
    void moveAddButton();

    QToolButton* button;
};
