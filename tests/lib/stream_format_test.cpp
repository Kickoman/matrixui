#include <doctest/doctest.h>

#include "core/lib/stream_format.h"

#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <stdexcept>


TEST_CASE("NullStream swallows everything written to it") {
    auto& sink = NullStream();

    sink << "text" << 42 << 3.5 << std::endl;

    CHECK(sink.good());
    // The same object comes back on every call, so a long-lived reference in a
    // trainer stays valid.
    CHECK(&NullStream() == &sink);
}

TEST_CASE("DefaultLogStream is std::cerr") {
    CHECK(&DefaultLogStream() == &std::cerr);
}

TEST_CASE("StreamFormatGuard restores flags, precision and fill") {
    std::ostringstream out;
    out << std::setfill('.') << std::setprecision(3) << std::hex;

    const auto flags = out.flags();
    const auto precision = out.precision();
    const auto fill = out.fill();

    {
        const StreamFormatGuard guard(out);
        out << std::scientific << std::setprecision(9) << std::setfill('*');
        CHECK(out.precision() == 9);
        CHECK(out.fill() == '*');
    }

    CHECK(out.flags() == flags);
    CHECK(out.precision() == precision);
    CHECK(out.fill() == fill);
}

TEST_CASE("StreamFormatGuard restores on the way out of a throw") {
    std::ostringstream out;
    const auto precision = out.precision();

    try {
        const StreamFormatGuard guard(out);
        out << std::setprecision(12);
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }

    CHECK(out.precision() == precision);
}
