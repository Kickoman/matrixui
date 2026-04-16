#include "gui/lib/mode_settings.h"


ModeSettings::ModeSettings(const QString& prefix)
    : prefix(prefix)
{
    qDebug() << "Settings in mode settings";
    auto keys = settings.allKeys();
    for (const auto& key : keys) {
        qDebug() << key;
    }
}

QVariant ModeSettings::getValue(const QString& name, const QVariant& defaultValue) {
    const auto newName = transformName(name);
    qDebug() << "Trying to get: " + newName;
    return settings.value(
        newName,
        defaultValue
    );
}

void ModeSettings::setValue(const QString& name, const QVariant& value) {
    settings.setValue(transformName(name), value);
}

QString ModeSettings::transformName(const QString& original) {
    return prefix + "_" + original;
}
