#include "main_window.h"
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QTimer>
#include <qgraphicsitem.h>
#include <qgraphicsscene.h>
#include <qnamespace.h>
#include <qpushbutton.h>
#include "digit_chart.h"
#include "paint_scene.h"
#include "digits_tester.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    // view = new QGraphicsView(this);
    // scene = new PaintScene(view);
    // view->setScene(scene);

    // openNetworkButton = new QPushButton("Open network", this);
    // clearButton = new QPushButton("Clear", this);
    // predictedNumber = new QLabel("0", this);
    // digitChart = new DigitChart(this);
    // QFont font = predictedNumber->font();
    // font.setPointSize(46);
    // predictedNumber->setFont(font);
    // predictedNumber->setAlignment(Qt::AlignCenter);

    // previewScene = new QGraphicsScene(this);
    // previewView = new QGraphicsView(this);
    // previewView->setScene(previewScene);
    // previewItem = new QGraphicsPixmapItem(QPixmap{QSize{28, 28}});
    // previewScene->addItem(previewItem);

    // auto vLayout = new QVBoxLayout();
    // auto hLayout = new QHBoxLayout();
    // auto buttonLayout = new QHBoxLayout();
    // auto resultLayout = new QVBoxLayout();
    // buttonLayout->addWidget(openNetworkButton);
    // resultLayout->addWidget(predictedNumber);
    // resultLayout->addWidget(digitChart);
    // resultLayout->addWidget(previewView);
    // vLayout->addLayout(buttonLayout);
    // vLayout->addLayout(hLayout);
    // hLayout->addWidget(view);
    // hLayout->addLayout(resultLayout);

    // auto* mainWidget = new QWidget(this);
    // mainWidget->setLayout(vLayout);
    // setCentralWidget(mainWidget);

    // scene->setSceneRect(0,0, view->width() - 20, view->height() - 20);

    // updateTimer = new QTimer(this);
    // connect(scene, &QGraphicsScene::changed, this, &MainWindow::handleSceneChanged);
    // connect(updateTimer, &QTimer::timeout, this, &MainWindow::handleUpdateTimer);

    view = new QGraphicsView(this);
    scene = new PaintScene(view);
    view->setScene(scene);

    openNetworkButton = new QPushButton("Open network", this);
    clearButton = new QPushButton("Clear", this);
    predictedNumber = new QLabel("0", this);
    digitChart = new DigitChart(this);
    QFont font = predictedNumber->font();
    font.setPointSize(46);
    predictedNumber->setFont(font);
    predictedNumber->setAlignment(Qt::AlignCenter);

    // Preview – use a separate scene and view
    previewScene = new QGraphicsScene(this);
    previewView = new QGraphicsView(this);
    previewView->setScene(previewScene);
    previewView->setFixedSize(300, 300);        // optional, make it visible
    previewView->setBackgroundRole(QPalette::Base);

    // Create preview item and add it to the preview scene
    previewItem = new QGraphicsPixmapItem();
    previewItem->setScale(5);
    previewScene->addItem(previewItem);
    // Optionally fit the view to the pixmap later

    // Layout (same as before, but previewView is already added to resultLayout)
    auto vLayout = new QVBoxLayout();
    auto hLayout = new QHBoxLayout();
    auto buttonLayout = new QHBoxLayout();
    auto resultLayout = new QVBoxLayout();
    buttonLayout->addWidget(openNetworkButton);
    buttonLayout->addWidget(clearButton);
    resultLayout->addWidget(predictedNumber);
    resultLayout->addWidget(digitChart);
    resultLayout->addWidget(previewView);
    vLayout->addLayout(buttonLayout);
    vLayout->addLayout(hLayout);
    hLayout->addWidget(view);
    hLayout->addLayout(resultLayout);

    auto* mainWidget = new QWidget(this);
    mainWidget->setLayout(vLayout);
    setCentralWidget(mainWidget);

    // Do NOT set a fixed sceneRect – we will use itemsBoundingRect() later
    scene->setSceneRect(0,0, view->width() - 20, view->height() - 20);

    updateTimer = new QTimer(this);
    connect(scene, &QGraphicsScene::changed, this, &MainWindow::handleSceneChanged);
    connect(updateTimer, &QTimer::timeout, this, &MainWindow::handleUpdateTimer);
    connect(clearButton, &QPushButton::clicked, this, &MainWindow::handleClearButton);
}

MainWindow::~MainWindow() {}


void MainWindow::setController(DigitsTester* controller) {
    this->controller = controller;
    connect(controller, &DigitsTester::infoUpdated, this, &MainWindow::handleInfoUpdated);
    connect(this, &MainWindow::sceneUpdated, controller, &DigitsTester::processUpdates);
    connect(this, &MainWindow::sceneUpdated, this, &MainWindow::handleSceneUpdated);
}


void MainWindow::resizeEvent(QResizeEvent *event)
{
    scene->setSceneRect(0,0, view->width() - 20, view->height() - 20);
    QWidget::resizeEvent(event);
}

void MainWindow::handleSceneChanged() {
    updateTimer->start(100);
}

QImage scaleKeepingAspectRatio(const QImage& src, const QSize& targetSize) {
    QImage dst(targetSize, QImage::Format_Grayscale8);
    dst.fill(Qt::white);   // padding color (black for MNIST style)

    QSize srcSize = src.size();
    double scale = qMin((double)targetSize.width() / srcSize.width(),
                        (double)targetSize.height() / srcSize.height());
    int newWidth = static_cast<int>(srcSize.width() * scale);
    int newHeight = static_cast<int>(srcSize.height() * scale);
    QImage scaled = src.scaled(newWidth, newHeight, Qt::KeepAspectRatio,
                               Qt::FastTransformation);

    // Center the scaled image
    int x = (targetSize.width() - newWidth) / 2;
    int y = (targetSize.height() - newHeight) / 2;
    QPainter painter(&dst);
    painter.drawImage(x, y, scaled);
    painter.setBrush(Qt::black);
    painter.end();

    return dst;
}

void centerOfMassAlign(QImage& img) {
    const int w = img.width();
    const int h = img.height();

    double sum = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    // Compute center of mass
    for (int y = 0; y < h; ++y) {
        const uchar* line = img.scanLine(y);
        for (int x = 0; x < w; ++x) {
            double mass = line[x] / 255.0; // assume white digit (after inversion)
            sum += mass;
            cx += x * mass;
            cy += y * mass;
        }
    }

    if (sum < 1e-6)
        return; // empty image

    cx /= sum;
    cy /= sum;

    // Desired center
    double targetX = (w - 1) / 2.0;
    double targetY = (h - 1) / 2.0;

    double dx = targetX - cx;
    double dy = targetY - cy;

    // Create shifted image
    QImage shifted(w, h, img.format());
    shifted.fill(0); // black background

    QPainter painter(&shifted);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
    painter.translate(dx, dy);
    painter.drawImage(0, 0, img);
    painter.end();

    img = shifted;
}

void MainWindow::handleUpdateTimer() {
    // QRectF sceneRect = scene->sceneRect();
    // if (sceneRect.isEmpty())
    //     sceneRect = scene->itemsBoundingRect();

    // int w = static_cast<int>(sceneRect.width());
    // int h = static_cast<int>(sceneRect.height());
    // if (w <= 0 || h <= 0)
    //     return;

    // // 2. Render to a grayscale image
    // QImage image(w, h, QImage::Format_Grayscale8);
    // image.fill(Qt::white);
    // QPainter painter(&image);
    // painter.setRenderHint(QPainter::Antialiasing);   // optional
    // painter.setRenderHint(QPainter::SmoothPixmapTransform);
    // scene->render(&painter, QRectF(), sceneRect);
    // painter.end();

    // image = image.scaled(
    //     28, 28, Qt::IgnoreAspectRatio, Qt::FastTransformation
    // );

    // emit sceneUpdated(image);


    /////////////////////////////////
    // Get the exact bounding rectangle of all items (the drawing)
    QRectF itemsRect = scene->itemsBoundingRect();
    qDebug() << "Rect: " << itemsRect.left() << " " << itemsRect.right() << " " << itemsRect.top() << " " << itemsRect.bottom();
    if (itemsRect.isEmpty())
        return;

    // Add a small margin (optional)
    itemsRect = itemsRect.adjusted(-10, -10, 10, 10);
    // Transform to a square
    const auto maxSide = std::max(itemsRect.width(), itemsRect.height());
    itemsRect.setWidth(maxSide);
    itemsRect.setHeight(maxSide);
    QPointF center = itemsRect.center();
    itemsRect.setSize(QSizeF(maxSide, maxSide));
    itemsRect.moveCenter(center);

    int w = static_cast<int>(itemsRect.width());
    int h = static_cast<int>(itemsRect.height());
    if (w <= 0 || h <= 0)
        return;

    // Render the cropped area to a grayscale image
    QImage image(w, h, QImage::Format_Grayscale8);
    image.fill(Qt::white);                       // white background
    QPainter painter(&image);
    // painter.setRenderHint(QPainter::Antialiasing);
    // painter.setRenderHint(QPainter::SmoothPixmapTransform);
    // Translate so that itemsRect maps to (0,0) in the image
    // painter.translate(-itemsRect.topLeft());
    scene->render(&painter, QRect(0, 0, w, h), itemsRect);
    painter.end();

    // Now scale to 28x28 while keeping aspect ratio and adding black borders
    QImage scaled = scaleKeepingAspectRatio(image, QSize(28, 28));
    scaled.invertPixels(QImage::InvertRgb);
    centerOfMassAlign(scaled);
    emit sceneUpdated(scaled);
}

void MainWindow::handleInfoUpdated(const DigitsTester::Info& info) {
    predictedNumber->setText(QString::number(info.predictedNumber));
    for (int i = 0; i < info.probabilities.size(); ++i) {
        digitChart->setValue(i, info.probabilities[i] * 100);
    }
}

void MainWindow::handleSceneUpdated(const QImage& image) {
    // QImage image(width, height, QImage::Format_Grayscale8);
    // for (int y = 0; y < height; ++y) {
    //     uchar* line = image.scanLine(y);
    //     for (int x = 0; x < width; ++x) {
    //         // matrix[y][x] true = white, false = black
    //         line[x] = matrix[y][x] ? 255 : 0;
    //     }
    // }
    // qDebug() << "Handle";
    // previewItem->setPixmap(QPixmap::fromImage(image));
        // Convert QImage to QPixmap and set on the preview item
    previewItem->setPixmap(QPixmap::fromImage(image));
    // Optionally fit the preview view to the pixmap
    previewView->fitInView(previewItem, Qt::KeepAspectRatio);
}

void MainWindow::handleClearButton() {
    scene->clear();
}
