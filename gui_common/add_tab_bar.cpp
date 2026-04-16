#include "gui_common/add_tab_bar.h"

AddTabBar::AddTabBar(QWidget* parent) : QTabBar(parent)
{ }

void AddTabBar::tabLayoutChange() {
    emit tabLayoutChanged();
}
