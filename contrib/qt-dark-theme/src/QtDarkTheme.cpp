#include "QtDarkTheme.h"

#include <QApplication>
#include <QFile>
#include <QString>

// Force the Qt resource initializer to run when this translation unit is linked.
// The rcc-generated init function is named after the resource file (qdarktheme).
extern void qInitResources_qdarktheme();

namespace QtDarkTheme {

void setup(QApplication& app, Theme theme)
{
    qInitResources_qdarktheme();

    const QString path = (theme == Theme::Dark)
        ? QStringLiteral(":/qdarktheme/dark.qss")
        : QStringLiteral(":/qdarktheme/light.qss");

    QFile file(path);
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    app.setStyleSheet(QString::fromUtf8(file.readAll()));
}

} // namespace QtDarkTheme
