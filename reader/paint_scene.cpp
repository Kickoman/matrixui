#include "paint_scene.h"
#include <qdebug.h>
#include <qgraphicsscene.h>
#include <qgraphicssceneevent.h>


constexpr int BRUSH_SIZE = 40;


PaintScene::PaintScene(QObject* parent)
    : QGraphicsScene(parent)
{}


PaintScene::~PaintScene()
{}

void PaintScene::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    // qDebug() << "mouse press";
    addEllipse(
        event->scenePos().x() - 5,
        event->scenePos().y() - 5,
        BRUSH_SIZE,
        BRUSH_SIZE,
        QPen(Qt::NoPen),
        QBrush(Qt::black)
    );
    previousPointPosition = event->scenePos();
}

void PaintScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    // qDebug() << "mouse move";
    addLine(previousPointPosition.x(),
            previousPointPosition.y(),
            event->scenePos().x(),
            event->scenePos().y(),
            QPen(Qt::black,BRUSH_SIZE,Qt::SolidLine,Qt::RoundCap)
    );
    previousPointPosition = event->scenePos();
}
