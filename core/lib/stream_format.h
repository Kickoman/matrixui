#pragma once

#include <ostream>

std::ostream& NullStream();

std::ostream& DefaultLogStream();

class StreamFormatGuard {
public:
    explicit StreamFormatGuard(std::ostream& stream)
        : stream(stream)
        , flags(stream.flags())
        , precision(stream.precision())
        , fill(stream.fill())
    {}

    ~StreamFormatGuard() {
        stream.flags(flags);
        stream.precision(precision);
        stream.fill(fill);
    }

    StreamFormatGuard(const StreamFormatGuard&) = delete;
    StreamFormatGuard& operator=(const StreamFormatGuard&) = delete;

private:
    std::ostream& stream;
    std::ios_base::fmtflags flags;
    std::streamsize precision;
    char fill;
};
