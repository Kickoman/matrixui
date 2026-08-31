#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace Tests {

// Redirects std::cout into a buffer for the duration of the scope.
//
// Needed because most of the words module still prints directly to std::cout;
// these captures are the characterization tests that pin the current output
// before the printing is moved behind std::ostream&.
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
