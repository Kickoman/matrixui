#include <doctest/doctest.h>

#include "core/lib/file_stream.h"
#include "core/lib/mapped_file.h"
#include "tests/support/temp_dir.h"

#include <cstring>
#include <utility>

using namespace Io;

TEST_CASE("MappedFile exposes the file's exact bytes") {
    const Tests::TempDir dir;
    const auto path = dir.write("payload.bin", "mapped contents");

    const MappedFile mapped(path);
    REQUIRE(mapped.getSize() == 15);
    CHECK(std::memcmp(mapped.getData(), "mapped contents", 15) == 0);
}

TEST_CASE("MappedFile is movable and the source is emptied") {
    const Tests::TempDir dir;
    MappedFile first(dir.write("payload.bin", "abc"));
    const auto* data = first.getData();

    MappedFile second(std::move(first));
    CHECK(second.getData() == data);
    CHECK(second.getSize() == 3);
    CHECK(first.getSize() == 0);
    CHECK(first.getData() == nullptr);

    MappedFile third;
    third = std::move(second);
    CHECK(third.getData() == data);
    CHECK(third.getSize() == 3);
    CHECK(second.getSize() == 0);
}

TEST_CASE("MappedFile maps an empty file with no data") {
    const Tests::TempDir dir;
    const MappedFile mapped(dir.write("empty.bin", ""));
    CHECK(mapped.getSize() == 0);
    CHECK(mapped.getData() == nullptr);
}

TEST_CASE("MappedFile rejects a missing file") {
    const Tests::TempDir dir;
    CHECK_THROWS_AS(MappedFile(dir.file("missing.bin")), Error);
}

TEST_CASE("Advice calls are harmless") {
    const Tests::TempDir dir;
    MappedFile mapped(dir.write("payload.bin", "abcd"));
    mapped.adviseWillNeed();
    mapped.adviseSequential();
    CHECK(mapped.getSize() == 4);
}

#ifdef __linux__
TEST_CASE("AvailableMemoryBytes reports a positive value") {
    CHECK(AvailableMemoryBytes() > 0);
}
#endif
