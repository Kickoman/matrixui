#include "pngreader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "matrix.h"

Matrix PngUtils::fromPNG(const std::string& filename) {
    int width, height, channels;
    unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1); // Force 1 channel

    if (!img) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    Matrix result(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char pixel = img[y * width + x];
            result(y, x) = (255.0 - pixel) / 255.0; // 0=white, 1=black
        }
    }

    stbi_image_free(img);
    return result;
}
