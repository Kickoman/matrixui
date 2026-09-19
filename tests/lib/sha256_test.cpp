#include <doctest/doctest.h>

#include "core/lib/sha256.h"

#include <sstream>
#include <string>

namespace {

std::string HexOf(const std::string& text) {
    return Hash::ToHex(Hash::Sha256Of(text.data(), text.size()));
}

}  // namespace

TEST_CASE("Sha256 matches the published vectors") {
    CHECK(HexOf("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(HexOf("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // 448 bits: one byte short of a block, so the length spills into a second one.
    CHECK(HexOf("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
          == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // 896 bits, and a million bytes for the multi-block path.
    CHECK(HexOf("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu")
          == "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    CHECK(HexOf(std::string(1000000, 'a'))
          == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("Sha256 does not care how the input is split") {
    const std::string text(200000, 'x');
    const auto whole = Hash::ToHex(Hash::Sha256Of(text.data(), text.size()));

    // Chunk sizes around the 64-byte block: under, exactly one, and over.
    for (const std::size_t step : {1u, 63u, 64u, 65u, 127u, 4096u}) {
        Hash::Sha256 hash;
        for (std::size_t offset = 0; offset < text.size(); offset += step) {
            hash.update(text.data() + offset, std::min(step, text.size() - offset));
        }
        CHECK(Hash::ToHex(hash.finish()) == whole);
    }
}

TEST_CASE("Sha256OfStream reads a stream to the end") {
    const std::string text(200000, 'q');
    std::istringstream in(text, std::ios::binary);

    CHECK(Hash::ToHex(Hash::Sha256OfStream(in))
          == Hash::ToHex(Hash::Sha256Of(text.data(), text.size())));

    std::istringstream empty("", std::ios::binary);
    CHECK(Hash::ToHex(Hash::Sha256OfStream(empty))
          == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("ToHex writes 64 lowercase hex digits") {
    const auto hex = Hash::ToHex(Hash::Sha256Of("", 0));
    CHECK(hex.size() == 64);
    CHECK(hex.find_first_not_of("0123456789abcdef") == std::string::npos);
}
