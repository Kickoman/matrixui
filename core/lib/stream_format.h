#pragma once

#include <ostream>

// A stream that discards everything written to it. The sink when verbose
// output is switched off.
std::ostream& NullStream();

// Where verbose output goes when no stream has been set. std::cerr, so a CLI
// that forgets to call setOutputStream still shows progress.
std::ostream& DefaultLogStream();

// Saves and restores a stream's formatting state. The printers set
// std::scientific, precision and fill without resetting them; on a stream
// shared with a GUI that would leak.
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
