#include "pngreader.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include "core/matrix/matrix.h"

Matrix PngUtils::fromImage(const std::string& filename, const unsigned targetHeight, const unsigned targetWidth) {
    int width, height, channels;
    unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1); // Force 1 channel

    if (!img) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    const bool resizeNecessary = targetHeight != static_cast<unsigned>(height)
                              || targetWidth != static_cast<unsigned>(width);

    unsigned char* resized = img;
    if (resizeNecessary) {
        resized = new unsigned char[targetHeight * targetWidth];
        stbir_resize_uint8_linear(img, width, height, 0, resized, targetWidth, targetHeight, 0, STBIR_1CHANNEL);
    }

    Matrix result(targetHeight, targetWidth);

    for (unsigned y = 0; y < targetHeight; y++) {
        for (unsigned x = 0; x < targetWidth; x++) {
            unsigned char pixel = resized[y * targetWidth + x];
            result(y, x) = (255.0 - pixel) / 255.0; // 0=white, 1=black
        }
    }

    stbi_image_free(img);
    if (resizeNecessary) {
        delete[] resized;
    }
    return result;
}


void PngUtils::toImage(const Matrix& image, const std::string& filename) {
    const int rows = static_cast<int>(image.getRows());
    const int cols = static_cast<int>(image.getCols());
    std::vector<unsigned char> pixels(rows * cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Invert: 0=white (255), 1=black (0), matching fromImage convention.
            const double v = std::clamp(image(y, x), 0.0, 1.0);
            pixels[y * cols + x] = static_cast<unsigned char>((1.0 - v) * 255.0);
        }
    }
    if (!stbi_write_png(filename.c_str(), cols, rows, 1, pixels.data(), cols)) {
        throw std::runtime_error("Failed to write image: " + filename);
    }
}
