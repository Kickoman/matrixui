#pragma once

#include "gui/lib/mode_widget.h"

#include <memory>
#include <ostream>

class AdvancedTerminal;
class WordsController;
class WordsDataTabWidget;
class WordsEvaluateTabWidget;
class WordsExploreTabWidget;
class QTabWidget;

class WordsModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit WordsModeWidget(QWidget* parent = nullptr);
    ~WordsModeWidget() override;

    void setController(WordsController* controller);
    std::ostream* getTerminalStream();

    void closeEvent(QCloseEvent* event) override;

public slots:
    void updateInfo();

private:
    std::unique_ptr<std::ostream> terminalStream;

    WordsController* controller = nullptr;

    QTabWidget* tabs = nullptr;
    WordsDataTabWidget* dataTab = nullptr;
    WordsExploreTabWidget* exploreTab = nullptr;
    WordsEvaluateTabWidget* evaluateTab = nullptr;
    AdvancedTerminal* terminal = nullptr;
};
