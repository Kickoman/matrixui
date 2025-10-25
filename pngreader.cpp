#include "pngreader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "matrix.h"

Matrix PngUtils::fromImage(const std::string& filename, const unsigned targetHeight, const unsigned targetWidth) {
    int width, height, channels;
    unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1); // Force 1 channel

    if (!img) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    const bool recizeNecessary = targetHeight != height || targetWidth != width;

    unsigned char* resized = img;  // by default, the image does not need recize
    if (recizeNecessary) {
        resized = new unsigned char[targetHeight * targetWidth];
        stbir_resize_uint8_linear(img, width, height, 0, resized, targetWidth, targetHeight, 0, STBIR_1CHANNEL);
    }

    Matrix result(targetHeight, targetWidth);

    for (int y = 0; y < targetHeight; y++) {
        for (int x = 0; x < targetWidth; x++) {
            unsigned char pixel = resized[y * width + x];
            result(y, x) = (255.0 - pixel) / 255.0; // 0=white, 1=black
        }
    }

    stbi_image_free(img);
    if (recizeNecessary) {
        delete resized;
    }
    return result;
}
