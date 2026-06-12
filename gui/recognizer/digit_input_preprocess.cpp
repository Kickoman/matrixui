#include "gui/recognizer/digit_input_preprocess.h"

#include <QPainter>
#include <QtGlobal>

namespace reader {

QImage scaleKeepingAspectRatio(const QImage& src, const QSize& targetSize) {
    QImage dst(targetSize, QImage::Format_Grayscale8);
    // Letterbox: pad with white to match white canvas before invert (same as training: white bg, dark stroke).
    dst.fill(Qt::white);

    const QSize srcSize = src.size();
    const double scale = qMin(static_cast<double>(targetSize.width()) / srcSize.width(),
                            static_cast<double>(targetSize.height()) / srcSize.height());
    const int newWidth = static_cast<int>(srcSize.width() * scale);
    const int newHeight = static_cast<int>(srcSize.height() * scale);
    const QImage scaled = src.scaled(newWidth, newHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    const int x = (targetSize.width() - newWidth) / 2;
    const int y = (targetSize.height() - newHeight) / 2;
    QPainter painter(&dst);
    painter.drawImage(x, y, scaled);
    painter.end();

    return dst;
}

void centerOfMassAlign(QImage& img) {
    const int w = img.width();
    const int h = img.height();

    double sum = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    // After invertPixels: background ~0, digit strokes brighter — weight by pixel intensity.
    for (int y = 0; y < h; ++y) {
        const uchar* line = img.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const double mass = line[x] / 255.0;
            sum += mass;
            cx += x * mass;
            cy += y * mass;
        }
    }

    if (sum < 1e-6)
        return;

    cx /= sum;
    cy /= sum;

    const double targetX = (w - 1) / 2.0;
    const double targetY = (h - 1) / 2.0;

    const double dx = targetX - cx;
    const double dy = targetY - cy;

    QImage shifted(w, h, img.format());
    shifted.fill(0);

    QPainter painter(&shifted);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
    painter.translate(dx, dy);
    painter.drawImage(0, 0, img);
    painter.end();

    img = shifted;
}

} // namespace reader
