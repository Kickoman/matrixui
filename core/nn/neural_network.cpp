#include "core/nn/neural_network.h"

#include "core/lib/text.h"


namespace Neural {

std::string LayersToTextRepresentation(const std::vector<std::size_t>& layers) {
    std::string text;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (i) {
            text += ", ";
        }
        text += std::to_string(layers[i]);
    }
    return text;
}

std::vector<std::size_t> TextRepresentationToLayers(const std::string& repr) {
    const auto cells = Text::Split(repr, ",");

    std::vector<std::size_t> result;
    result.reserve(cells.size());
    for (const auto& cell : cells) {
        if (const auto value = Text::ParseNumber<std::size_t>(cell)) {
            result.push_back(*value);
        }
    }
    return result;
}

}
