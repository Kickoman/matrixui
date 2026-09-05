#include "cli/lib/png_reader.h"

#include "core/lib/cache.h"
#include "core/png/pngreader.h"

#include <memory>

namespace CliLib {

PngReader MakeCachedPngReader(const std::size_t imageWidth, const std::size_t imageHeight) {
    auto pngCache = std::make_shared<cache::LRUCache<std::filesystem::path, Matrix>>();

    return [pngCache, imageWidth, imageHeight](const std::filesystem::path& path) {
        if (const auto cached = pngCache->get(path); cached.has_value()) {
            return *cached;
        }

        const auto result = PngUtils::fromImage(path.string(), imageHeight, imageWidth)
            .transform(1, imageHeight * imageWidth);
        pngCache->put(path, result);
        return result;
    };
}

}  // namespace CliLib
