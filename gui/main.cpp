#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QFontDatabase>
#include <QTimer>
#include "main_window.h"
#include "main_controller.h"

#include "gui/lib/theme.h"

extern void qInitResources_fonts();

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

    // Headless smoke check: bring the window up, save what it rendered, and quit.
    // Works under QT_QPA_PLATFORM=offscreen, so "does the GUI actually start on
    // this machine?" can be answered over SSH, with no window server involved.
    QCommandLineOption screenshot(
        QStringList() << "screenshot",
        "Render the main window to <file> as PNG, then exit",
        "file"
    );
    parser.addOption(screenshot);
    parser.process(a);

    if ((!parser.isSet(theme) && AppTheme::IsSystemDarkMode())
        || (parser.isSet(theme) && parser.value(theme) == "dark")
    ) {
        AppTheme::SetTheme(Theme::Dark);
        AppTheme::ApplyTheme(a);
    }

    constexpr const char* fontPath = ":/fonts/ubuntu-sans.ttf";
    const auto fontId = QFontDatabase::addApplicationFont(fontPath);
    if (fontId != -1) {
        qDebug() << "Font ID installed: " << fontId;
        // Ask Qt what the family is actually called rather than assuming the file
        // name matches it -- the resolved name differs across font backends.
        const auto families = QFontDatabase::applicationFontFamilies(fontId);
        if (!families.isEmpty()) {
            a.setFont(QFont(families.first()));
        }
    } else {
        qDebug() << "Failed to load fonts...";
    }

    MainController controller;
    MainWindow window;
    window.setController(&controller);
    window.handleNewTabRequested();
    window.show();

    if (parser.isSet(screenshot)) {
        const QString target = parser.value(screenshot);
        int status = 0;
        // Give the platform plugin an event loop turn or two to actually paint;
        // grabbing before the window is exposed yields a blank image.
        QTimer::singleShot(1500, &a, [&a, &window, &target, &status]() {
            if (window.grab().save(target, "PNG")) {
                qInfo() << "Screenshot written to" << target;
            } else {
                qWarning() << "Failed to write screenshot to" << target;
                status = 1;
            }
            a.quit();
        });
        a.exec();
        return status;
    }

    return a.exec();
}
