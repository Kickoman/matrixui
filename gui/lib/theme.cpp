#include "gui/lib/theme.h"

#include "QtDarkTheme.h"

// Source - https://stackoverflow.com/a/78854851
// Posted by Nick Bolton
// Retrieved 2026-04-17, License - CC BY-SA 4.0

#include <QGuiApplication>
#include <QApplication>
#include <QPalette>
#include <QStyleHints>
#include <QtCharts/QtCharts>


namespace {
#ifdef __linux__
static std::optional<bool> queryGSettingsDark() {
    // Check color-scheme first
    FILE* pipe = popen(
        "gsettings get org.gnome.desktop.interface color-scheme 2>/dev/null", "r");
    if (pipe) {
        char buf[64] = {};
        fgets(buf, sizeof(buf), pipe);
        pclose(pipe);
        std::string val(buf);
        if (val.find("prefer-dark")  != std::string::npos) return true;
        if (val.find("prefer-light") != std::string::npos) return false;
    }

    // Ubuntu 24.04 fallback: check gtk-theme
    pipe = popen(
        "gsettings get org.gnome.desktop.interface gtk-theme 2>/dev/null", "r");
    if (pipe) {
        char buf[64] = {};
        fgets(buf, sizeof(buf), pipe);
        pclose(pipe);
        std::string val(buf);
        // Convert to lowercase for comparison
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        if (val.find("dark") != std::string::npos) return true;
        return false;
    }

    return std::nullopt; // gsettings not available
}
#endif
}

bool AppTheme::IsSystemDarkMode() {
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    const auto scheme = QApplication::styleHints()->colorScheme();
    if (scheme != Qt::ColorScheme::Unknown) {
        return scheme == Qt::ColorScheme::Dark;
    }
#endif

#ifdef __linux__
    if (const auto result = queryGSettingsDark()) {
        return *result;
    }
#endif

    const QPalette defaultPalette;
    const auto text   = defaultPalette.color(QPalette::WindowText);
    const auto window = defaultPalette.color(QPalette::Window);
    return text.lightness() > window.lightness();
}


Theme AppTheme::theme = Theme::Default;

void AppTheme::SetTheme(const Theme theme) {
    AppTheme::theme = theme;
}

Theme AppTheme::GetTheme() {
    return theme;
}

void AppTheme::ApplyTheme(QApplication& app) {
    switch (theme) {
        case Theme::Default:
            app.setStyleSheet({});
            break;
        case Theme::Dark:
            QtDarkTheme::setup(app, QtDarkTheme::Theme::Dark);
            break;
    }
}

void AppTheme::ApplyTheme(QChart* chart) {
    switch (theme) {
        case Theme::Default:
            chart->setTheme(QChart::ChartThemeQt);
            break;
        case Theme::Dark:
            chart->setTheme(QChart::ChartThemeDark);
            break;
    }
}
