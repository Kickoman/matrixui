#pragma once

#include <ostream>

namespace Words {

// A stream that discards everything written to it.
//
// Used as the sink when verbose output is switched off. Note the classifier's
// trainer does NOT do this -- core/classifier/trainer.cpp falls back to
// std::cerr when !verbose, so setVerbose(false) there still prints. That is a
// bug worth not copying: under a GUI it would leak training chatter to the
// terminal the application was launched from.
std::ostream& NullStream();

// Where verbose output goes when no stream has been set. std::cerr, so a CLI
// that forgets to call setOutputStream still shows progress.
std::ostream& DefaultLogStream();

// Saves and restores a stream's formatting state.
//
// Every printer here used to run against a fresh std::cout, so leaked
// manipulators were invisible. Once the same stream is shared with the rest of
// a GUI they become real bugs -- the training progress line, for instance, sets
// std::scientific and never restores it.
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

}  // namespace Words
