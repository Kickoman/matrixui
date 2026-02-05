#ifndef PNGREADER_H
#define PNGREADER_H

#include "matrix.h"
#include "utils/cache.h"

namespace PngUtils {

using Cache = ::cache::LRUCache<std::string, Matrix>;

Matrix fromImage(
    const std::string& filename,
    const unsigned targetHeight,
    const unsigned targetWidth
);


Matrix fromImage(
    const std::string& filename,
    const unsigned targetHeight,
    const unsigned targetWidth,
    Cache& cache
);

}

#endif // PNGREADER_H
