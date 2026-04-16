#pragma once


#include <QSettings>


class ModeSettings {
public:
    explicit ModeSettings(const QString& prefix);

    QVariant getValue(const QString& name, const QVariant& defaultValue = {});
    void setValue(const QString& name, const QVariant& value);

private:
    QString transformName(const QString& original);

    const QString prefix;
    QSettings settings;
};
