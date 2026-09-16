#pragma once

#include <QPointF>
#include <QVector>
#include <QWidget>

#include <functional>
#include <vector>


class FunctionPlotWidget : public QWidget
{
    Q_OBJECT
public:
    struct Curve {
        QString label;
        std::function<double(double)> evaluate;
    };

    explicit FunctionPlotWidget(QWidget* parent = nullptr);

    void setPoints(const QVector<QPointF>& value);
    void setCurves(std::vector<Curve> value);
    void setEditable(bool value);

    QSize minimumSizeHint() const override;

public slots:
    void resetView();

signals:
    void pointAdded(double x, double y);
    void pointMoved(int index, double x, double y);
    void pointDeleted(int index);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    enum class Drag { None, Pending, Pan, Point };

    struct Ticks {
        double first;
        double step;
    };

    QRectF plotRect() const;
    double toPixelX(double x) const;
    double toPixelY(double y) const;
    double toDataX(double px) const;
    double toDataY(double py) const;
    int hitTest(const QPointF& position) const;
    static Ticks NiceTicks(double minimum, double maximum, int targetCount);
    static QString FormatTick(double value);
    bool isDarkPalette() const;
    QColor curveColor(int index) const;

    void drawGrid(QPainter& painter);
    void drawAxes(QPainter& painter);
    void drawCurves(QPainter& painter);
    void drawPoints(QPainter& painter);
    void drawLegend(QPainter& painter);

    QVector<QPointF> points;
    std::vector<Curve> curves;
    bool editable = true;

    double xMin = -10;
    double xMax = 10;
    double yMin = -10;
    double yMax = 10;

    Drag drag = Drag::None;
    QPoint pressPosition;
    double pressXMin = 0;
    double pressXMax = 0;
    double pressYMin = 0;
    double pressYMax = 0;
    int activeIndex = -1;
    int hoverIndex = -1;
};
