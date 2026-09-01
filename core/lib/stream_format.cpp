#include "core/lib/stream_format.h"

#include <iostream>
#include <streambuf>

namespace {

class NullBuffer : public std::streambuf {
protected:
    int_type overflow(const int_type ch) override { return ch; }
    std::streamsize xsputn(const char_type*, const std::streamsize count) override { return count; }
};

}  // namespace

std::ostream& NullStream() {
    static NullBuffer buffer;
    static std::ostream stream(&buffer);
    return stream;
}

std::ostream& DefaultLogStream() {
    return std::cerr;
}
