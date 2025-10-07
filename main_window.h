#pragma once

#include <QMainWindow>
#include <qmainwindow.h>
#include <qtmetamacros.h>


class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow() = default;
};
