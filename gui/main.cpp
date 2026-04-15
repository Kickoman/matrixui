#include <QApplication>
#include <QSettings>
#include "main_window.h"
#include "main_controller.h"

#include <QSettings>

int main(int argc, char** argv) {
    QApplication a(argc, argv);
    a.setApplicationName("Networks");
    a.setApplicationDisplayName("Neural Networks by Kastus");
    a.setOrganizationName("Kastus");

    MainController controller;
    MainWindow window;
    window.setController(&controller);
    window.handleNewTabRequested();
    window.show();
    return a.exec();
}
