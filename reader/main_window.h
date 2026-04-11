#pragma once

#include <QImage>
#include <QMainWindow>
#include "digits_tester.h"

class QGraphicsScene;
class QGraphicsView;
class QPushButton;
class QLabel;
class DigitChart;
class QGraphicsPixmapItem;
class QTimer;
class QResizeEvent;


class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    void setController(DigitsTester* tester);
    void resizeEvent(QResizeEvent* event) override;

signals:
    void sceneUpdated(const QImage& image);

private slots:
    void handleInfoUpdated(const DigitsTester::Info& info);
    void handleSceneChanged();
    void handleUpdateTimer();
    void handleSceneUpdated(const QImage& image);
    void handleClearButton();
    void handleOpenNetworkClicked();

private:
    QTimer* updateTimer = nullptr;
    DigitsTester* controller = nullptr;

    QGraphicsScene* scene = nullptr;
    QGraphicsView* view = nullptr;

    QGraphicsScene* previewScene = nullptr;
    QGraphicsView* previewView = nullptr;
    QGraphicsPixmapItem* previewItem = nullptr;

    QPushButton* openNetworkButton = nullptr;
    QPushButton* clearButton = nullptr;
    QLabel* predictedNumber = nullptr;
    DigitChart* digitChart = nullptr;
};
