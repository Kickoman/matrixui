#ifndef PNGREADER_H
#define PNGREADER_H

#include "matrix/matrix.h"

namespace PngUtils {

Matrix fromImage(
    const std::string& filename,
    const unsigned targetHeight,
    const unsigned targetWidth
);

// Write a grayscale matrix (values in [0,1], 0=white 1=black) to a PNG file.
// The matrix is treated as (rows x cols) pixels.
void toImage(const Matrix& image, const std::string& filename);

}

#endif // PNGREADER_H
