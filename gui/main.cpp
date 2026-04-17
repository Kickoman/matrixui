#include <QApplication>
#include <QCommandLineParser>
#include "main_window.h"
#include "main_controller.h"

#include "gui/lib/theme.h"

int main(int argc, char** argv) {
    QApplication a(argc, argv);
    a.setApplicationName("Networks");
    a.setApplicationDisplayName("Neural Networks by Kastus");
    a.setOrganizationName("Kastus");

    QCommandLineParser parser;
    parser.setApplicationDescription("Neural networks playground");
    parser.addHelpOption();

    QCommandLineOption theme(
        QStringList() << "t" << "theme",
        "Set force application style (dark/light)",
        "theme"
    );
    parser.addOption(theme);
    parser.process(a);

    if (!parser.isSet(theme) && AppTheme::IsSystemDarkMode()
        || parser.isSet(theme) && parser.value(theme) == "dark"
    ) {
        AppTheme::SetTheme(Theme::Dark);
        AppTheme::ApplyTheme(a);
    }

    MainController controller;
    MainWindow window;
    window.setController(&controller);
    window.handleNewTabRequested();
    window.show();
    return a.exec();
}
