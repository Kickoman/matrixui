#ifndef PNGREADER_H
#define PNGREADER_H

#include "matrix.h"

namespace PngUtils {

Matrix fromImage(const std::string& filename, const unsigned targetHeight, const unsigned targetWidth);

}

#endif // PNGREADER_H
