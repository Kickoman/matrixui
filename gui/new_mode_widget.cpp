#include "new_mode_widget.h"
#include "mode_factory.h"

#include <QPushButton>
#include <QVBoxLayout>

NewModeWidget::NewModeWidget(QWidget* parent)
    : ModeWidget("New mode", parent)
{
    auto* digitsClassifierButton = new QPushButton("Digits classifier", this);
    auto* digitsRecognitionPlaygroundButton = new QPushButton("Digits recognition playground", this);
    auto* ganButton = new QPushButton("GAN generative network training", this);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(digitsClassifierButton);
    layout->addWidget(digitsRecognitionPlaygroundButton);
    layout->addWidget(ganButton);

    setLayout(layout);

    connect(
        digitsClassifierButton,
        &QPushButton::clicked,
        [this]{ emit modeRequested(ModeType::DigitsClassifier); }
    );
    connect(
        digitsRecognitionPlaygroundButton,
        &QPushButton::clicked,
        [this]{ emit modeRequested(ModeType::DigitsRecognizer); }
    );
    connect(
        ganButton,
        &QPushButton::clicked,
        [this]{ emit modeRequested(ModeType::DigitsGenerator); }
    );
}
