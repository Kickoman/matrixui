#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace Tests {

// Redirects std::cout into a buffer for the duration of the scope.
class CoutCapture {
public:
    CoutCapture() : original(std::cout.rdbuf(buffer.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(original); }

    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;

    std::string str() const { return buffer.str(); }

private:
    std::ostringstream buffer;
    std::streambuf* original;
};

}  // namespace Tests
