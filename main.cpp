#include <QApplication>
#include "main_window.h"


int main(int argc, char** argv) {
    QApplication::setStyle("windows");
    QApplication a(argc, argv);
    MainWindow window;
    window.show();
    return a.exec();
}
