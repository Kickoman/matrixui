#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace Hash {

using Sha256Digest = std::array<std::uint8_t, 32>;

// Incremental SHA-256 (FIPS 180-4). Feed it with update(), read it once with
// finish(); the object is spent afterwards.
class Sha256 {
public:
    void update(const void* data, std::size_t size);
    Sha256Digest finish();

private:
    void compress(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<std::uint8_t, 64> buffer{};
    std::size_t buffered = 0;
    std::uint64_t totalBytes = 0;
};

Sha256Digest Sha256Of(const void* data, std::size_t size);

// Reads to the end of the stream in fixed-size chunks, so a file never has to
// be held whole.
Sha256Digest Sha256OfStream(std::istream& in);

std::string ToHex(const Sha256Digest& digest);

}  // namespace Hash
