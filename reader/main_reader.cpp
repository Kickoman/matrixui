#include "main_window.h"
#include "digits_tester.h"

#include <QApplication>

int main(int argc, char** argv) {
    QApplication a(argc, argv);
    MainWindow window;

    DigitsTester tester;
    tester.loadNetwork("/home/kanstancin/Documents/projects/matrixgui/build/tryagain1.wgt");
    window.setController(&tester);

    window.show();
    return a.exec();
}
