#include "mode_container_widget.h"

#include "new_mode_widget.h"

#include <QStackedLayout>

ModeContainerWidget::ModeContainerWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* greetingsWidget = new NewModeWidget(this);
    mainLayout = new QStackedLayout(this);
    mainLayout->addWidget(greetingsWidget);

    connect(greetingsWidget, &NewModeWidget::modeRequested, [this](const ModeType mode){
        emit modeRequested(this, mode);
    });
}

const QString& ModeContainerWidget::getModeName() const {
    auto* modeWidget = qobject_cast<ModeWidget*>(mainLayout->currentWidget());
    if (!modeWidget) {
        qDebug() << "Can't retrieve mode name: mode widget is not ModeWidget somehow";
        return {};
    }
    return modeWidget->getModeName();
}

void ModeContainerWidget::setModeWidget(ModeWidget* modeWidget) {
    modeWidget->setParent(this);
    auto* oldWidget = mainLayout->currentWidget();
    mainLayout->addWidget(modeWidget);
    mainLayout->setCurrentWidget(modeWidget);
    mainLayout->removeWidget(oldWidget);
    oldWidget->deleteLater();
    emit modeNameChanged(modeWidget->getModeName());
}
