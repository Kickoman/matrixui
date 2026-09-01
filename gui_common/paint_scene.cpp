#include "gui_common/paint_scene.h"

#include <QGraphicsSceneMouseEvent>


namespace {
constexpr int BRUSH_SIZE = 40;
constexpr double BRUSH_RADIUS = BRUSH_SIZE / 2.0;
}  // namespace

PaintScene::PaintScene(QObject* parent)
    : QGraphicsScene(parent)
{}

PaintScene::~PaintScene() = default;

void PaintScene::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    const QPointF p = event->scenePos();
    addEllipse(
        p.x() - BRUSH_RADIUS,
        p.y() - BRUSH_RADIUS,
        BRUSH_SIZE,
        BRUSH_SIZE,
        QPen(Qt::NoPen),
        QBrush(Qt::black));
    previousPointPosition = p;
}

void PaintScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    addLine(previousPointPosition.x(),
            previousPointPosition.y(),
            event->scenePos().x(),
            event->scenePos().y(),
            QPen(Qt::black, BRUSH_SIZE, Qt::SolidLine, Qt::RoundCap));
    previousPointPosition = event->scenePos();
}
