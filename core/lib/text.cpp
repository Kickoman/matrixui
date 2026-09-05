#include "core/lib/text.h"

#include <algorithm>
#include <cctype>
#include <sstream>

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> SplitWords(const std::string& text) {
    std::vector<std::string> result;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        result.push_back(ToLower(word));
    }
    return result;
}
