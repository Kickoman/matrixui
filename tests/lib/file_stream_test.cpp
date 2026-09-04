#include <doctest/doctest.h>

#include "core/lib/file_stream.h"
#include "tests/support/temp_dir.h"

#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

TEST_CASE("OpenForReading throws with the path when the file is missing") {
    const Tests::TempDir dir;
    const auto missing = dir.file("absent.txt");

    CHECK_THROWS_AS(Io::OpenForReading(missing), Io::Error);
    try {
        Io::OpenForReading(missing);
    } catch (const Io::Error& error) {
        CHECK(std::string(error.what()).find(missing.string()) != std::string::npos);
    }
}

TEST_CASE("OpenForWriting throws with the path when the target is unwritable") {
    const Tests::TempDir dir;
    const auto unwritable = dir.file("no-such-directory/out.txt");

    CHECK_THROWS_AS(Io::OpenForWriting(unwritable), Io::Error);
}

TEST_CASE("TryOpenForReading is empty for a missing file") {
    const Tests::TempDir dir;
    CHECK_FALSE(Io::TryOpenForReading(dir.file("absent.txt")).has_value());
    CHECK(Io::TryOpenForReading(dir.write("here.txt", "x")).has_value());
}

TEST_CASE("ReadFile hands the stream to the reader") {
    const Tests::TempDir dir;
    const auto path = dir.write("words.txt", "alpha beta");

    const auto first = Io::ReadFile(path, [](std::istream& in) {
        std::string word;
        in >> word;
        return word;
    });

    CHECK(first == "alpha");
}

TEST_CASE("ReadFile tags an error from the reader with the file name") {
    const Tests::TempDir dir;
    const auto path = dir.write("payload.bin", "garbage");

    try {
        Io::ReadFile(path, [](std::istream&) -> int { throw std::runtime_error("bad header"); });
        FAIL("expected a throw");
    } catch (const Io::Error& error) {
        CHECK(std::string(error.what()) == "bad header: " + path.string());
    }
}

TEST_CASE("TryReadFile is empty for a missing file but still tags reader errors") {
    const Tests::TempDir dir;

    CHECK_FALSE(Io::TryReadFile(dir.file("absent.bin"),
                                [](std::istream&) { return 1; }).has_value());

    const auto path = dir.write("present.bin", "x");
    CHECK_THROWS_AS(
        Io::TryReadFile(path, [](std::istream&) -> int { throw std::runtime_error("nope"); }),
        Io::Error);
}

TEST_CASE("WriteFile creates the file and the writer fills it") {
    const Tests::TempDir dir;
    const auto path = dir.file("out.txt");

    Io::WriteFile(path, [](std::ostream& out) { out << "written"; });

    const auto text = Io::ReadFile(path, [](std::istream& in) {
        std::string word;
        in >> word;
        return word;
    });
    CHECK(text == "written");
}
