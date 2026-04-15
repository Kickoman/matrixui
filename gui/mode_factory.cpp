#include "mode_factory.h"

#include "digits_classifier_controller.h"
#include "digits_classifier_mode_widget.h"
#include "digits_recognizer_mode_widget.h"
#include "digits_recognizer_mode_controller.h"

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
}


Mode CreateMode(ModeType mode) {
    switch (mode) {
        case ModeType::DigitsClassifier:
            return CreateDigitsClassifierMode();
        case ModeType::DigitsRecognizer:
            return CreateDigitsRecognizerMode();
        default:
            throw std::runtime_error("Unsupported");
    }
}
