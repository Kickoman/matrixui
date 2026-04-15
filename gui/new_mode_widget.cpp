#include "new_mode_widget.h"
#include "mode_factory.h"

#include <QHBoxLayout>
#include <QToolButton>
#include <qtoolbutton.h>

NewModeWidget::NewModeWidget(QWidget* parent)
    : ModeWidget("New mode", parent)
{
    auto* digitsClassifierButton = new QToolButton(this);
    auto* digitsRecognitionPlaygroundButton = new QToolButton(this);
    auto* ganButton = new QToolButton(this);

    digitsClassifierButton->setText("Digits\nclassifier");
    digitsRecognitionPlaygroundButton->setText("Digits\nrecognition\nplayground");
    ganButton->setText("GAN\ngenerative\nnetwork\ntraining");

    const QSize buttonSize(300, 300);
    const QFont buttonFont("sans", 24);
    digitsClassifierButton->setFixedSize(buttonSize);
    digitsClassifierButton->setFont(buttonFont);
    digitsRecognitionPlaygroundButton->setFixedSize(buttonSize);
    digitsRecognitionPlaygroundButton->setFont(buttonFont);
    ganButton->setFixedSize(buttonSize);
    ganButton->setFont(buttonFont);
    ganButton->setDisabled(true);

    auto* layout = new QHBoxLayout(this);
    layout->addWidget(digitsClassifierButton);
    layout->addWidget(digitsRecognitionPlaygroundButton);
    layout->addWidget(ganButton);

    setLayout(layout);

    connect(
        digitsClassifierButton,
        &QToolButton::clicked,
        [this]{ emit modeRequested(ModeType::DigitsClassifier); }
    );
    connect(
        digitsRecognitionPlaygroundButton,
        &QToolButton::clicked,
        [this]{ emit modeRequested(ModeType::DigitsRecognizer); }
    );
    connect(
        ganButton,
        &QToolButton::clicked,
        [this]{ emit modeRequested(ModeType::DigitsGenerator); }
    );
}
