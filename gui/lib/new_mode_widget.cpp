#include "gui/lib/new_mode_widget.h"
#include "gui/lib/mode_factory.h"

#include <QGridLayout>
#include <QToolButton>
#include <qtoolbutton.h>

NewModeWidget::NewModeWidget(QWidget* parent)
    : ModeWidget("New mode", parent)
{
    auto* digitsClassifierButton = new QToolButton(this);
    auto* digitsRecognitionPlaygroundButton = new QToolButton(this);
    auto* ganButton = new QToolButton(this);
    auto* wordsButton = new QToolButton(this);
    auto* functionsButton = new QToolButton(this);

    digitsClassifierButton->setText("Digits\nclassifier");
    digitsRecognitionPlaygroundButton->setText("Digits\nrecognition\nplayground");
    ganButton->setText("GAN\ngenerative\nnetwork\ntraining");
    wordsButton->setText("Word\nembeddings\n(SGNS)");
    functionsButton->setText("Functions\nsymbolic\nregression");

    const QSize buttonSize(300, 300);
    const QFont buttonFont("sans", 24);
    for (auto* button : {digitsClassifierButton, digitsRecognitionPlaygroundButton,
                         ganButton, wordsButton, functionsButton}) {
        button->setFixedSize(buttonSize);
        button->setFont(buttonFont);
    }

    // A 3x2 grid: three fixed 300px buttons in a row need ~940px, which fits;
    // a fourth per row would overflow anything narrower than ~1260px. The sixth
    // cell is free, a seventh mode means rethinking this again.
    auto* layout = new QGridLayout(this);
    layout->addWidget(digitsClassifierButton, 0, 0);
    layout->addWidget(digitsRecognitionPlaygroundButton, 0, 1);
    layout->addWidget(ganButton, 0, 2);
    layout->addWidget(wordsButton, 1, 0);
    layout->addWidget(functionsButton, 1, 1);
    layout->setAlignment(Qt::AlignCenter);

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
    connect(
        wordsButton,
        &QToolButton::clicked,
        [this]{ emit modeRequested(ModeType::Words); }
    );
    connect(
        functionsButton,
        &QToolButton::clicked,
        [this]{ emit modeRequested(ModeType::Functions); }
    );
}
