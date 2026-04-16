#pragma once

#include <QImage>
#include <QSize>

namespace reader {

inline constexpr int kMnistSize = 28;
inline constexpr int kMnistPixels = kMnistSize * kMnistSize;

/** Total margin subtracted from view width/height when setting the scene rect (padding around drawable area). */
inline constexpr int kSceneViewChromePx = 20;

/**
 * Per-side padding added around the tight ink bounding box before squaring to a square.
 * MNIST-style digits occupy roughly 20×20 in a 28×28 frame (4px margin per side on the
 * full image, or equivalently ~20% of the 20px content span per side). Using max(w,h) of
 * the tight bbox scales margin with stroke size instead of fixed pixels.
 */
inline constexpr double kCropMarginFractionOfMaxSide = 4.0 / 20.0;

inline constexpr int kUpdateDebounceMs = 100;

inline constexpr int kPreviewViewSizePx = 300;
inline constexpr double kPreviewPixmapScale = 5.0;

QImage scaleKeepingAspectRatio(const QImage& src, const QSize& targetSize);
void centerOfMassAlign(QImage& img);

} // namespace reader
