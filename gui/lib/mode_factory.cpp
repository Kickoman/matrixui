#include "gui/lib/mode_factory.h"

#include "gui/classifier/digits_classifier_controller.h"
#include "gui/classifier/digits_classifier_mode_widget.h"
#include "gui/recognizer/digits_recognizer_mode_widget.h"
#include "gui/recognizer/digits_recognizer_mode_controller.h"
#include "gui/generator/digits_generator_controller.h"
#include "gui/generator/digits_generator_mode_widget.h"
#include "gui/words/words_controller.h"
#include "gui/words/words_mode_widget.h"
#include "gui/functions/functions_controller.h"
#include "gui/functions/functions_mode_widget.h"

#include <stdexcept>


namespace {
    Mode CreateDigitsClassifierMode() {
        auto* controller = new DigitsClassifierController();
        auto* view = new DigitsClassifierModeWidget();
        controller->setLogger(view->getTerminalStream());
        view->setController(controller);
        return {
            .controller = controller,
            .view = view,
        };
    }

    Mode CreateDigitsRecognizerMode() {
        auto* controller = new DigitsRecognizerModeController();
        auto* view = new DigitsRecognizerModeWidget();
        view->setController(controller);
        return {
            .controller = controller,
            .view = view,
        };
    }

    Mode CreateWordsMode() {
        auto* controller = new WordsController();
        auto* view = new WordsModeWidget();
        controller->setLogger(view->getTerminalStream());
        view->setController(controller);
        return {
            .controller = controller,
            .view = view,
        };
    }

    Mode CreateFunctionsMode() {
        auto* controller = new FunctionsController();
        auto* view = new FunctionsModeWidget();
        controller->setLogger(view->getTerminalStream());
        view->setController(controller);
        return {
            .controller = controller,
            .view = view,
        };
    }

    Mode CreateDigitsGeneratorMode() {
        auto* controller = new DigitsGeneratorController();
        auto* view = new DigitsGeneratorModeWidget();
        controller->setLogger(view->getTerminalStream());
        view->setController(controller);
        return {
            .controller = controller,
            .view = view,
        };
    }
}


Mode CreateMode(ModeType mode) {
    switch (mode) {
        case ModeType::DigitsClassifier:
            return CreateDigitsClassifierMode();
        case ModeType::DigitsRecognizer:
            return CreateDigitsRecognizerMode();
        case ModeType::DigitsGenerator:
            return CreateDigitsGeneratorMode();
        case ModeType::Words:
            return CreateWordsMode();
        case ModeType::Functions:
            return CreateFunctionsMode();
    }
    throw std::runtime_error("Unsupported mode");
}
