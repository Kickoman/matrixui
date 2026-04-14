#pragma once

#include <QWidget>


class ModeWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ModeWidget(const QString& modeName, QWidget* parent = nullptr);

    QString getModeName() const;

private:
    const QString modeName;
};
