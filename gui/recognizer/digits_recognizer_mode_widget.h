#pragma once

#include "gui/lib/mode_widget.h"
#include "gui/recognizer/digits_recognizer_mode_controller.h"

class QImage;
class QGraphicsScene;
class QGraphicsView;
class QPushButton;
class QLabel;
class DigitChart;
class QGraphicsPixmapItem;
class QTimer;
class QResizeEvent;
class DigitsRecognizerModeController;


class DigitsRecognizerModeWidget : public ModeWidget
{
    Q_OBJECT
public:
    explicit DigitsRecognizerModeWidget(QWidget* parent = nullptr);
    ~DigitsRecognizerModeWidget();

    void setController(DigitsRecognizerModeController* tester);
    void resizeEvent(QResizeEvent* event) override;

signals:
    void sceneUpdated(const QImage& image);

private slots:
    void handleInfoUpdated(const DigitsRecognizerModeController::Info& info);
    void handleSceneChanged();
    void handleUpdateTimer();
    void handleSceneUpdated(const QImage& image);
    void handleClearButton();
    void handleOpenNetworkClicked();

private:
    QTimer* updateTimer = nullptr;
    DigitsRecognizerModeController* controller = nullptr;

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
