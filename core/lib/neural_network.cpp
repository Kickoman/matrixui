#include "core/lib/neural_network.h"
#include <charconv>


namespace {

constexpr std::string_view WHITESPACES = " \t\n\r\f\v";

constexpr std::string_view Trim(std::string_view sv) {
    const auto first = sv.find_first_not_of(WHITESPACES);
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = sv.find_last_not_of(WHITESPACES);
    return sv.substr(first, last - first + 1);
}

}


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
    std::vector<std::size_t> result;
    result.reserve(static_cast<std::size_t>(std::ranges::count(repr, ',')) + 1);

    for (auto part : std::views::split(repr, ',')) {
        const std::string_view token = Trim(std::string_view(part.begin(), part.end()));
        if (token.empty()) {
            continue;
        }

        std::size_t value{};
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec == std::errc{} && ptr == token.data() + token.size()) {
            result.push_back(value);
        }
    }
    return result;
}

}
