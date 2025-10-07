#include "main_window.h"

#include <QLabel>


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    auto* label = new QLabel(this);
    label->setText("Hello Liza!");

    setCentralWidget(label);
}
