#include "main_window.h"
#include "digits_tester.h"

#include <QApplication>
#include <QString>
#include <iostream>

int main(int argc, char** argv) {
    QApplication a(argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "MatrixGui_reader") << " <network.wgt>\n";
        return 1;
    }

    DigitsClassifierModeWidget window;

    DigitsTester tester;
    if (!tester.loadNetwork(QString::fromUtf8(argv[1]))) {
        std::cerr << "Failed to load network: " << argv[1] << '\n';
        return 1;
    }
    window.setController(&tester);

    window.show();
    return a.exec();
}
