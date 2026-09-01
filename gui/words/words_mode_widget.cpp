#include "gui/words/words_mode_widget.h"

#include "gui/words/words_controller.h"
#include "gui/words/words_data_tab_widget.h"
#include "gui/words/words_evaluate_tab_widget.h"
#include "gui/words/words_explore_tab_widget.h"
#include "gui_common/advanced_terminal.h"

#include <QTabWidget>
#include <QVBoxLayout>

WordsModeWidget::WordsModeWidget(QWidget* parent)
    : ModeWidget("Words", parent)
{
    terminal = new AdvancedTerminal(this);
    terminal->setMaximumHeight(250);

    dataTab = new WordsDataTabWidget(this);
    exploreTab = new WordsExploreTabWidget(this);
    evaluateTab = new WordsEvaluateTabWidget(this);

    tabs = new QTabWidget(this);
    tabs->addTab(dataTab, "Data && Training");
    tabs->addTab(exploreTab, "Explore");
    tabs->addTab(evaluateTab, "Evaluate");

    // The terminal sits below the tabs: shared by all three and always visible.
    auto* root = new QVBoxLayout();
    root->addWidget(tabs, 1);
    root->addWidget(terminal);
    setLayout(root);
}

WordsModeWidget::~WordsModeWidget()
{
    if (controller) {
        controller->saveSettings();
        controller->requestStop();
    }
}

void WordsModeWidget::setController(WordsController* newController)
{
    controller = newController;
    controller->loadSettings();

    dataTab->setController(controller);
    exploreTab->setController(controller);
    evaluateTab->setController(controller);

    connect(controller, &WordsController::infoUpdated, this, &WordsModeWidget::updateInfo);
    connect(controller, &WordsController::trainProgressed,
            dataTab, &WordsDataTabWidget::handleTrainProgress);

    updateInfo();
}

std::ostream* WordsModeWidget::getTerminalStream()
{
    if (!terminalStream) {
        terminalStream = createTerminalOStream(terminal);
    }
    return terminalStream.get();
}

void WordsModeWidget::updateInfo()
{
    if (!controller) {
        return;
    }
    const auto info = controller->getInfo();
    dataTab->updateInfo(info);
    exploreTab->updateInfo(info);
    evaluateTab->updateInfo(info);
}

void WordsModeWidget::closeEvent(QCloseEvent* event)
{
    if (controller) {
        controller->saveSettings();
        controller->requestStop();
    }
    QWidget::closeEvent(event);
}
