#pragma once

#include <QPixmap>
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

    struct SampleKey {
        double xMin = 0;
        double xMax = 0;
        double left = 0;
        double right = 0;
        quint64 generation = 0;

        bool operator==(const SampleKey& other) const = default;
    };

    struct LayerKey {
        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        int width = 0;
        int height = 0;
        quint64 curveGeneration = 0;
        quint64 pointGeneration = 0;
        bool dark = false;

        bool operator==(const LayerKey& other) const = default;
    };

    LayerKey currentLayerKey() const;
    void rebuildLayer(const LayerKey& key);
    void resampleCurves(const QRectF& rect);

    QVector<QPointF> points;
    std::vector<Curve> curves;
    quint64 curveGeneration = 0;
    quint64 pointGeneration = 0;
    QPixmap layer;
    LayerKey layerKey;
    SampleKey samplesKey;
    std::vector<std::vector<double>> samples;
    std::vector<QString> legendLabels;
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
