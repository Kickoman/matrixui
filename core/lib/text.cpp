#include "core/lib/text.h"

#include <algorithm>
#include <cctype>

namespace Text {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

void Utf8CharacterOffsets(const std::string_view text, std::vector<std::size_t>& offsets) {
    offsets.clear();
    std::size_t index = 0;
    while (index < text.size()) {
        offsets.push_back(index);
        ++index;
        while (index < text.size() && IsUtf8Continuation(static_cast<unsigned char>(text[index]))) {
            ++index;
        }
    }
    offsets.push_back(text.size());
}

std::vector<std::size_t> Utf8CharacterOffsets(const std::string_view text) {
    std::vector<std::size_t> offsets;
    Utf8CharacterOffsets(text, offsets);
    return offsets;
}

std::string_view Trim(const std::string_view text, const std::string_view whitespace) {
    const auto begin = text.find_first_not_of(whitespace);
    if (begin == std::string_view::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> Split(const std::string_view text, const std::string_view separators) {
    std::vector<std::string> cells;
    std::size_t begin = 0;
    while (true) {
        const auto separator = text.find_first_of(separators, begin);
        cells.emplace_back(text.substr(begin, separator - begin));
        if (separator == std::string_view::npos) {
            return cells;
        }
        begin = separator + 1;
    }
}

std::vector<std::string> SplitWords(const std::string_view text) {
    std::vector<std::string> words;
    for (auto& token : Split(text, kWhitespace)) {
        if (!token.empty()) {
            words.push_back(ToLower(std::move(token)));
        }
    }
    return words;
}

}  // namespace Text
