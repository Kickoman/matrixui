#pragma once

class QApplication;

namespace QtDarkTheme {

enum class Theme {
    Dark,
    Light,
};

void setup(QApplication& app, Theme theme = Theme::Dark);

} // namespace QtDarkTheme
