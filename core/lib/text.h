#pragma once

#include <string>
#include <vector>

// ASCII case folding. Bytes outside a-z/A-Z are left as they are, so UTF-8
// sequences pass through unchanged rather than being mangled per byte.
std::string ToLower(std::string value);

// Splits on any whitespace and lowercases each token. Leading, trailing and
// repeated separators produce no empty entries.
std::vector<std::string> SplitWords(const std::string& text);
