#pragma once

#include <charconv>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace Text {

inline constexpr std::string_view kWhitespace = " \t\n\r\f\v";

// ascii-only. utf-8 would be unchanged.
std::string ToLower(std::string value);

inline bool IsUtf8Continuation(const unsigned char byte) {
    return (byte & 0xC0) == 0x80;
}

// byte offsets where each utf-8 character starts, plus text.size() as the last
// entry, so character i spans [offsets[i], offsets[i + 1]). a stray
// continuation byte starts a character of its own rather than running past the
// end.
std::vector<std::size_t> Utf8CharacterOffsets(std::string_view text);
void Utf8CharacterOffsets(std::string_view text, std::vector<std::size_t>& offsets);

// a view into `text`, so the caller keeps `text` alive
std::string_view Trim(std::string_view text, std::string_view whitespace = kWhitespace);

// split by any of `separators`, keeping empty cells: "a,,b" is three cells, so
// a row keeps its column count. cells come out as they are -- Trim them when
// the format allows padding around them.
std::vector<std::string> Split(std::string_view text, std::string_view separators);

// split by whitespace and lowercase every token: runs of whitespace collapse
// and no token comes out empty. what a vocabulary lookup wants.
std::vector<std::string> SplitWords(std::string_view text);

// nullopt unless the whole text, once trimmed, is one number. a leading '+' is
// allowed, which std::from_chars rejects on its own.
template <typename T>
std::optional<T> ParseNumber(std::string_view text) {
    std::string_view number = Trim(text);
    if (number.empty()) {
        return std::nullopt;
    }
    if (number.front() == '+') {
        number.remove_prefix(1);
        if (number.empty() || number.front() == '-') {
            return std::nullopt;
        }
    }

    T value{};
    const auto [ptr, ec] = std::from_chars(number.data(), number.data() + number.size(), value);
    if (ec != std::errc{} || ptr != number.data() + number.size()) {
        return std::nullopt;
    }
    return value;
}

}  // namespace Text
