#include "gui/recognizer/digits_recognizer_mode_widget.h"

#include "gui_common/digit_chart.h"
#include "gui_common/paint_scene.h"

#include "gui/lib/mode_widget.h"

#include "gui/recognizer/digit_input_preprocess.h"
#include "gui/recognizer/digits_recognizer_mode_controller.h"

#include <QFileDialog>
#include <QFont>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QPalette>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

DigitsRecognizerModeWidget::DigitsRecognizerModeWidget(QWidget* parent)
    : ModeWidget("Digits recognizer", parent)
{
    view = new QGraphicsView(this);
    scene = new PaintScene(view);
    view->setScene(scene);

    openNetworkButton = new QPushButton(tr("Open network"), this);
    clearButton = new QPushButton(tr("Clear"), this);
    predictedNumber = new QLabel(QStringLiteral("0"), this);
    digitChart = new DigitChart(this);
    QFont font = predictedNumber->font();
    font.setPointSize(46);
    predictedNumber->setFont(font);
    predictedNumber->setAlignment(Qt::AlignCenter);

    previewScene = new QGraphicsScene(this);
    previewView = new QGraphicsView(this);
    previewView->setScene(previewScene);
    previewView->setFixedSize(reader::kPreviewViewSizePx, reader::kPreviewViewSizePx);
    previewView->setBackgroundRole(QPalette::Base);

    previewItem = new QGraphicsPixmapItem();
    previewItem->setScale(reader::kPreviewPixmapScale);
    previewScene->addItem(previewItem);

    auto* vLayout = new QVBoxLayout();
    auto* hLayout = new QHBoxLayout();
    auto* buttonLayout = new QHBoxLayout();
    auto* resultLayout = new QVBoxLayout();
    buttonLayout->addWidget(openNetworkButton);
    buttonLayout->addWidget(clearButton);
    resultLayout->addWidget(predictedNumber);
    resultLayout->addWidget(digitChart);
    resultLayout->addWidget(previewView);
    vLayout->addLayout(buttonLayout);
    vLayout->addLayout(hLayout);
    hLayout->addWidget(view);
    hLayout->addLayout(resultLayout);

    setLayout(vLayout);

    scene->setSceneRect(0, 0, view->width() - reader::kSceneViewChromePx,
                        view->height() - reader::kSceneViewChromePx);

    updateTimer = new QTimer(this);
    connect(scene, &QGraphicsScene::changed, this, &DigitsRecognizerModeWidget::handleSceneChanged);
    connect(updateTimer, &QTimer::timeout, this, &DigitsRecognizerModeWidget::handleUpdateTimer);
    connect(clearButton, &QPushButton::clicked, this, &DigitsRecognizerModeWidget::handleClearButton);
}

DigitsRecognizerModeWidget::~DigitsRecognizerModeWidget() = default;

void DigitsRecognizerModeWidget::setController(DigitsRecognizerModeController* controller) {
    this->controller = controller;
    connect(controller, &DigitsRecognizerModeController::infoUpdated, this, &DigitsRecognizerModeWidget::handleInfoUpdated);
    connect(this, &DigitsRecognizerModeWidget::sceneUpdated, controller, &DigitsRecognizerModeController::processUpdates);
    connect(this, &DigitsRecognizerModeWidget::sceneUpdated, this, &DigitsRecognizerModeWidget::handleSceneUpdated);
    connect(openNetworkButton, &QPushButton::clicked, this, &DigitsRecognizerModeWidget::handleOpenNetworkClicked);
}

void DigitsRecognizerModeWidget::resizeEvent(QResizeEvent* event) {
    scene->setSceneRect(0, 0, view->width() - reader::kSceneViewChromePx,
                        view->height() - reader::kSceneViewChromePx);
    ModeWidget::resizeEvent(event);
}

void DigitsRecognizerModeWidget::handleSceneChanged() {
    updateTimer->start(reader::kUpdateDebounceMs);
}

void DigitsRecognizerModeWidget::handleUpdateTimer() {
    QRectF itemsRect = scene->itemsBoundingRect();
    if (itemsRect.isEmpty())
        return;

    const double tightSpan = std::max(itemsRect.width(), itemsRect.height());
    const double cropMargin = tightSpan * reader::kCropMarginFractionOfMaxSide;
    itemsRect = itemsRect.adjusted(-cropMargin, -cropMargin, cropMargin, cropMargin);
    const auto maxSide = std::max(itemsRect.width(), itemsRect.height());
    itemsRect.setWidth(maxSide);
    itemsRect.setHeight(maxSide);
    const QPointF center = itemsRect.center();
    itemsRect.setSize(QSizeF(maxSide, maxSide));
    itemsRect.moveCenter(center);

    const int w = static_cast<int>(itemsRect.width());
    const int h = static_cast<int>(itemsRect.height());
    if (w <= 0 || h <= 0)
        return;

    QImage image(w, h, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    QPainter painter(&image);
    scene->render(&painter, QRect(0, 0, w, h), itemsRect);
    painter.end();

    QImage scaled = reader::scaleKeepingAspectRatio(
        image, QSize(reader::kMnistSize, reader::kMnistSize));
    scaled.invertPixels(QImage::InvertRgb);
    reader::centerOfMassAlign(scaled);
    emit sceneUpdated(scaled);
}

void DigitsRecognizerModeWidget::handleInfoUpdated(const DigitsRecognizerModeController::Info& info) {
    predictedNumber->setText(QString::number(info.predictedNumber));
    for (int i = 0; i < info.probabilities.size(); ++i) {
        digitChart->setValue(static_cast<unsigned>(i), info.probabilities[i] * 100);
    }
}

void DigitsRecognizerModeWidget::handleSceneUpdated(const QImage& image) {
    previewItem->setPixmap(QPixmap::fromImage(image));
    previewView->fitInView(previewItem, Qt::KeepAspectRatio);
}

void DigitsRecognizerModeWidget::handleClearButton() {
    scene->clear();
}

void DigitsRecognizerModeWidget::handleOpenNetworkClicked() {
    if (!controller)
        return;
    const QString path = QFileDialog::getOpenFileName(
        this,
        tr("Open network"),
        QString(),
        tr("Weight files (*.wgt);;All files (*)"));
    if (path.isEmpty())
        return;
    try {
        if (!controller->loadNetwork(path)) {
            QMessageBox::warning(this, tr("Open network"),
                                 tr("Failed to load network:\n%1").arg(path));
        }
    } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Open network"),
                             tr("Failed to load network:\n%1\n\n%2")
                                 .arg(path, QString::fromUtf8(e.what())));
    }
}
