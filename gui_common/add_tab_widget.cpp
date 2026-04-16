#include "gui_common/add_tab_widget.h"
#include "gui_common/add_tab_bar.h"

#include <QTabBar>
#include <QToolButton>
#include <QTimer>

AddTabWidget::AddTabWidget(QWidget* parent) : QTabWidget(parent)
{
    auto* bar = new AddTabBar(this);
    setTabBar(bar);

    button = new QToolButton(this);
    button->setText("+");
    button->setAutoRaise(true);
    button->resize(button->sizeHint());

    connect(button, &QToolButton::clicked, this, &AddTabWidget::newTabRequested);
    connect(bar, &AddTabBar::tabLayoutChanged, this, [this]() {
        QTimer::singleShot(0, this, [this]() { moveAddButton(); });
    });
}

void AddTabWidget::showEvent(QShowEvent* e) {
    QTabWidget::showEvent(e);
    QTimer::singleShot(0, this, [this]() {
        button->show();
        button->raise();
        moveAddButton();
    });
}

void AddTabWidget::resizeEvent(QResizeEvent* e) {
    QTabWidget::resizeEvent(e);
    moveAddButton();
}

void AddTabWidget::moveAddButton() {
    QTabBar* bar = tabBar();
    int n = bar->count();
    if (n == 0) return;

    QRect barGeom = bar->geometry();
    QRect lastTab = bar->tabRect(n - 1);

    int x = barGeom.left() + lastTab.right() + 2;
    int y = barGeom.top() + (barGeom.height() - button->height()) / 2;
    button->move(x, y);
}
