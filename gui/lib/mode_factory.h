#pragma once


class ModeController;
class ModeWidget;


enum class ModeType {
    DigitsClassifier,
    DigitsRecognizer,
    DigitsGenerator,
};


struct Mode {
    ModeController* controller;
    ModeWidget* view;
};


Mode CreateMode(ModeType mode);
