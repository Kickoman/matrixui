#pragma once


enum class Theme {
  Default,
  Dark
};

class QChart;
class QApplication;

class AppTheme {
public:
  AppTheme() = delete;

  static bool IsSystemDarkMode();

  static void SetTheme(Theme theme);
  static Theme GetTheme();

  static void ApplyTheme(QApplication& app);
  static void ApplyTheme(QChart* chart);

private:
  static Theme theme;
};
