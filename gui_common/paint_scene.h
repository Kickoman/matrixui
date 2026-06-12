#pragma once

#include <QGraphicsScene>


class PaintScene : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit PaintScene(QObject* parent = nullptr);
    ~PaintScene();

private:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

    QPointF previousPointPosition;
};
