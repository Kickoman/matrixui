#pragma once


class ModeController;
class ModeWidget;


enum class ModeType {
    DigitsClassifier,
    DigitsRecognizer,
    DigitsGenerator,
    Words,
    Functions,
};


struct Mode {
    ModeController* controller;
    ModeWidget* view;
};


Mode CreateMode(ModeType mode);
