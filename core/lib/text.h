#pragma once

#include <string>
#include <vector>

// ascii-only. utf-8 would be unchanged.
std::string ToLower(std::string value);

// split by whitespace
std::vector<std::string> SplitWords(const std::string& text);
