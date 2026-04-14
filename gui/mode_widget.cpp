#include "mode_widget.h"


ModeWidget::ModeWidget(const QString& name, QWidget* parent)
    : QWidget(parent)
    , modeName(name)
{ }

QString ModeWidget::getModeName() const {
    return modeName;
}
