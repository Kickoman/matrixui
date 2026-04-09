#include "main_window.h"
#include "digits_tester.h"

#include <QApplication>
#include <iostream>

int main(int argc, char** argv) {
    QApplication a(argc, argv);

    if (argc < 2) {
        std::cerr << "Specify network" << std::endl;
    }
    MainWindow window;

    DigitsTester tester;
    tester.loadNetwork(argv[1]);
    window.setController(&tester);

    window.show();
    return a.exec();
}
