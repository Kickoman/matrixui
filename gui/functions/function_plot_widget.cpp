#include "gui/functions/function_plot_widget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kHitRadiusPx = 8;
constexpr int kClickThresholdPx = 4;
constexpr int kSampleStepPx = 2;
constexpr int kMarginLeftPx = 56;
constexpr int kMarginBottomPx = 24;
constexpr int kMarginTopPx = 8;
constexpr int kMarginRightPx = 12;
constexpr double kMinSpan = 1e-9;
constexpr double kMaxSpan = 1e12;

// Hues far enough apart to stay distinguishable in a legend of ten.
constexpr int kCurveHues[] = {210, 0, 130, 40, 280, 170, 60, 320, 190, 100};

}  // namespace


FunctionPlotWidget::FunctionPlotWidget(QWidget* parent)
    : QWidget(parent)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setAutoFillBackground(false);
}

QSize FunctionPlotWidget::minimumSizeHint() const {
    return {320, 240};
}

void FunctionPlotWidget::setPoints(const QVector<QPointF>& value) {
    points = value;
    if (hoverIndex >= points.size()) {
        hoverIndex = -1;
    }
    update();
}

void FunctionPlotWidget::setCurves(std::vector<Curve> value) {
    curves = std::move(value);
    update();
}

void FunctionPlotWidget::setEditable(const bool value) {
    editable = value;
    if (!editable) {
        hoverIndex = -1;
        unsetCursor();
    }
    update();
}

void FunctionPlotWidget::resetView() {
    if (points.isEmpty()) {
        xMin = -10;
        xMax = 10;
        yMin = -10;
        yMax = 10;
        update();
        return;
    }

    auto left = points.front().x();
    auto right = left;
    auto bottom = points.front().y();
    auto top = bottom;
    for (const auto& point : points) {
        left = std::min(left, point.x());
        right = std::max(right, point.x());
        bottom = std::min(bottom, point.y());
        top = std::max(top, point.y());
    }

    const auto pad = [](double& low, double& high) {
        const auto span = high - low;
        if (span < kMinSpan) {
            low -= 5;
            high += 5;
            return;
        }
        low -= span * 0.1;
        high += span * 0.1;
    };
    pad(left, right);
    pad(bottom, top);

    xMin = left;
    xMax = right;
    yMin = bottom;
    yMax = top;
    update();
}

QRectF FunctionPlotWidget::plotRect() const {
    return QRectF(kMarginLeftPx, kMarginTopPx,
                  std::max(1, width() - kMarginLeftPx - kMarginRightPx),
                  std::max(1, height() - kMarginTopPx - kMarginBottomPx));
}

double FunctionPlotWidget::toPixelX(const double x) const {
    const auto rect = plotRect();
    return rect.left() + (x - xMin) / (xMax - xMin) * rect.width();
}

double FunctionPlotWidget::toPixelY(const double y) const {
    const auto rect = plotRect();
    return rect.bottom() - (y - yMin) / (yMax - yMin) * rect.height();
}

double FunctionPlotWidget::toDataX(const double px) const {
    const auto rect = plotRect();
    return xMin + (px - rect.left()) / rect.width() * (xMax - xMin);
}

double FunctionPlotWidget::toDataY(const double py) const {
    const auto rect = plotRect();
    return yMin + (rect.bottom() - py) / rect.height() * (yMax - yMin);
}

int FunctionPlotWidget::hitTest(const QPointF& position) const {
    int best = -1;
    double bestDistance = kHitRadiusPx * kHitRadiusPx;
    for (int i = 0; i < points.size(); ++i) {
        const auto dx = toPixelX(points[i].x()) - position.x();
        const auto dy = toPixelY(points[i].y()) - position.y();
        const auto distance = dx * dx + dy * dy;
        if (distance <= bestDistance) {
            bestDistance = distance;
            best = i;
        }
    }
    return best;
}

FunctionPlotWidget::Ticks FunctionPlotWidget::NiceTicks(
    const double minimum, const double maximum, const int targetCount)
{
    const auto range = maximum - minimum;
    if (!(range > 0) || targetCount <= 0) {
        return {minimum, 1};
    }
    const auto rough = range / targetCount;
    const auto magnitude = std::pow(10.0, std::floor(std::log10(rough)));
    const auto normalized = rough / magnitude;
    const auto multiplier = normalized < 1.5 ? 1.0
                          : normalized < 3.0 ? 2.0
                          : normalized < 7.0 ? 5.0
                                             : 10.0;
    const auto step = multiplier * magnitude;
    return {std::ceil(minimum / step) * step, step};
}

QString FunctionPlotWidget::FormatTick(const double value) {
    return QString::number(std::abs(value) < 1e-12 ? 0.0 : value, 'g', 6);
}

bool FunctionPlotWidget::isDarkPalette() const {
    return palette().color(QPalette::Base).lightness() < 128;
}

QColor FunctionPlotWidget::curveColor(const int index) const {
    const auto hue = kCurveHues[index % static_cast<int>(std::size(kCurveHues))];
    return QColor::fromHsv(hue, 200, isDarkPalette() ? 235 : 170);
}

void FunctionPlotWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), palette().color(QPalette::Base));

    painter.save();
    painter.setClipRect(plotRect());
    drawGrid(painter);
    drawCurves(painter);
    drawPoints(painter);
    painter.restore();

    drawAxes(painter);
    drawLegend(painter);
}

void FunctionPlotWidget::drawGrid(QPainter& painter) {
    const auto rect = plotRect();
    auto gridColor = palette().color(QPalette::Mid);
    gridColor.setAlpha(70);
    painter.setPen(QPen(gridColor, 1));

    const auto xTicks = NiceTicks(xMin, xMax, std::max(2, static_cast<int>(rect.width() / 80)));
    for (auto x = xTicks.first; x <= xMax; x += xTicks.step) {
        const auto px = toPixelX(x);
        painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()));
    }
    const auto yTicks = NiceTicks(yMin, yMax, std::max(2, static_cast<int>(rect.height() / 50)));
    for (auto y = yTicks.first; y <= yMax; y += yTicks.step) {
        const auto py = toPixelY(y);
        painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py));
    }

    auto axisColor = palette().color(QPalette::Mid);
    axisColor.setAlpha(180);
    painter.setPen(QPen(axisColor, 1.5));
    if (xMin < 0 && xMax > 0) {
        const auto px = toPixelX(0);
        painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()));
    }
    if (yMin < 0 && yMax > 0) {
        const auto py = toPixelY(0);
        painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py));
    }
}

void FunctionPlotWidget::drawAxes(QPainter& painter) {
    const auto rect = plotRect();
    painter.setPen(QPen(palette().color(QPalette::WindowText), 1));
    painter.drawRect(rect);

    const auto metrics = painter.fontMetrics();
    const auto xTicks = NiceTicks(xMin, xMax, std::max(2, static_cast<int>(rect.width() / 80)));
    for (auto x = xTicks.first; x <= xMax; x += xTicks.step) {
        const auto text = FormatTick(x);
        const auto px = toPixelX(x);
        painter.drawText(QPointF(px - metrics.horizontalAdvance(text) / 2.0,
                                 rect.bottom() + metrics.ascent() + 4),
                         text);
    }

    const auto yTicks = NiceTicks(yMin, yMax, std::max(2, static_cast<int>(rect.height() / 50)));
    for (auto y = yTicks.first; y <= yMax; y += yTicks.step) {
        const auto text = FormatTick(y);
        const auto py = toPixelY(y);
        painter.drawText(QPointF(rect.left() - metrics.horizontalAdvance(text) - 6,
                                 py + metrics.ascent() / 2.0 - 1),
                         text);
    }
}

void FunctionPlotWidget::drawCurves(QPainter& painter) {
    const auto rect = plotRect();
    const auto limit = rect.height() * 10;

    for (std::size_t i = 0; i < curves.size(); ++i) {
        const auto& curve = curves[i];
        if (!curve.evaluate) {
            continue;
        }
        painter.setPen(QPen(curveColor(static_cast<int>(i)), 2));

        QPolygonF segment;
        const auto flush = [&painter, &segment] {
            if (segment.size() >= 2) {
                painter.drawPolyline(segment);
            }
            segment.clear();
        };

        for (auto px = rect.left(); px <= rect.right(); px += kSampleStepPx) {
            const auto value = curve.evaluate(toDataX(px));
            if (!std::isfinite(value)) {
                flush();
                continue;
            }
            const auto py = toPixelY(value);
            if (!std::isfinite(py) || py < rect.top() - limit || py > rect.bottom() + limit) {
                flush();
                continue;
            }
            segment.append(QPointF(px, py));
        }
        flush();
    }
}

void FunctionPlotWidget::drawPoints(QPainter& painter) {
    const auto base = palette().color(QPalette::Base);
    for (int i = 0; i < points.size(); ++i) {
        const QPointF position(toPixelX(points[i].x()), toPixelY(points[i].y()));
        const auto active = (i == hoverIndex) || (drag == Drag::Point && i == activeIndex);
        const auto radius = active ? 6.0 : 4.0;
        painter.setPen(QPen(base, 1.5));
        painter.setBrush(active ? palette().color(QPalette::Highlight)
                                : palette().color(QPalette::WindowText));
        painter.drawEllipse(position, radius, radius);
    }
    painter.setBrush(Qt::NoBrush);
}

void FunctionPlotWidget::drawLegend(QPainter& painter) {
    if (curves.empty()) {
        return;
    }

    const auto metrics = painter.fontMetrics();
    const auto rowHeight = metrics.height() + 4;
    constexpr int swatchWidth = 16;
    constexpr int padding = 8;

    int textWidth = 0;
    std::vector<QString> labels;
    labels.reserve(curves.size());
    for (const auto& curve : curves) {
        auto label = metrics.elidedText(curve.label, Qt::ElideRight, 260);
        textWidth = std::max(textWidth, metrics.horizontalAdvance(label));
        labels.push_back(std::move(label));
    }

    const QRectF box(plotRect().left() + padding, plotRect().top() + padding,
                     swatchWidth + 6 + textWidth + 2 * padding,
                     static_cast<int>(curves.size()) * rowHeight + padding);
    auto background = palette().color(QPalette::Window);
    background.setAlpha(215);
    painter.setPen(QPen(palette().color(QPalette::Mid), 1));
    painter.setBrush(background);
    painter.drawRoundedRect(box, 4, 4);
    painter.setBrush(Qt::NoBrush);

    for (std::size_t i = 0; i < labels.size(); ++i) {
        const auto y = box.top() + padding / 2.0 + (static_cast<int>(i) + 0.5) * rowHeight;
        painter.setPen(QPen(curveColor(static_cast<int>(i)), 3));
        painter.drawLine(QPointF(box.left() + padding, y),
                         QPointF(box.left() + padding + swatchWidth, y));
        painter.setPen(palette().color(QPalette::WindowText));
        painter.drawText(QPointF(box.left() + padding + swatchWidth + 6,
                                 y + metrics.ascent() / 2.0 - 1),
                         labels[i]);
    }
}

void FunctionPlotWidget::mousePressEvent(QMouseEvent* event) {
    const auto position = event->position();
    if (!plotRect().contains(position)) {
        return;
    }

    if (event->button() == Qt::RightButton) {
        if (editable && drag == Drag::None) {
            if (const auto index = hitTest(position); index >= 0) {
                emit pointDeleted(index);
            }
        }
        return;
    }
    if (event->button() != Qt::LeftButton) {
        return;
    }

    activeIndex = hitTest(position);
    pressPosition = event->pos();
    pressXMin = xMin;
    pressXMax = xMax;
    pressYMin = yMin;
    pressYMax = yMax;
    drag = Drag::Pending;
}

void FunctionPlotWidget::mouseMoveEvent(QMouseEvent* event) {
    const auto position = event->position();

    if (drag == Drag::None) {
        const auto index = editable ? hitTest(position) : -1;
        if (index != hoverIndex) {
            hoverIndex = index;
            if (hoverIndex >= 0) {
                setCursor(Qt::PointingHandCursor);
            } else {
                unsetCursor();
            }
            update();
        }
        return;
    }

    if (drag == Drag::Pending) {
        if ((event->pos() - pressPosition).manhattanLength() <= kClickThresholdPx) {
            return;
        }
        drag = (editable && activeIndex >= 0) ? Drag::Point : Drag::Pan;
    }

    if (drag == Drag::Pan) {
        const auto rect = plotRect();
        const auto dx = (event->pos().x() - pressPosition.x()) / rect.width() * (pressXMax - pressXMin);
        const auto dy = (event->pos().y() - pressPosition.y()) / rect.height() * (pressYMax - pressYMin);
        xMin = pressXMin - dx;
        xMax = pressXMax - dx;
        yMin = pressYMin + dy;
        yMax = pressYMax + dy;
        update();
        return;
    }

    if (drag == Drag::Point && activeIndex >= 0 && activeIndex < points.size()) {
        emit pointMoved(activeIndex, toDataX(position.x()), toDataY(position.y()));
    }
}

void FunctionPlotWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }
    const auto wasPending = drag == Drag::Pending;
    drag = Drag::None;

    if (!wasPending) {
        activeIndex = -1;
        update();
        return;
    }

    if (editable && activeIndex < 0 && plotRect().contains(event->position())) {
        emit pointAdded(toDataX(event->position().x()), toDataY(event->position().y()));
    }
    activeIndex = -1;
}

void FunctionPlotWidget::wheelEvent(QWheelEvent* event) {
    const auto position = event->position();
    if (!plotRect().contains(position)) {
        return;
    }

    const auto steps = event->angleDelta().y() / 120.0;
    if (steps == 0) {
        return;
    }
    const auto factor = std::pow(1.25, steps);

    const auto anchorX = toDataX(position.x());
    const auto anchorY = toDataY(position.y());
    const auto newXMin = anchorX - (anchorX - xMin) / factor;
    const auto newXMax = anchorX + (xMax - anchorX) / factor;
    const auto newYMin = anchorY - (anchorY - yMin) / factor;
    const auto newYMax = anchorY + (yMax - anchorY) / factor;

    const auto xSpan = newXMax - newXMin;
    const auto ySpan = newYMax - newYMin;
    if (xSpan < kMinSpan || xSpan > kMaxSpan || ySpan < kMinSpan || ySpan > kMaxSpan) {
        return;
    }

    xMin = newXMin;
    xMax = newXMax;
    yMin = newYMin;
    yMax = newYMax;
    event->accept();
    update();
}

void FunctionPlotWidget::leaveEvent(QEvent*) {
    if (hoverIndex != -1) {
        hoverIndex = -1;
        unsetCursor();
        update();
    }
}
