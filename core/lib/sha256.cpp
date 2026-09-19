#include "core/lib/sha256.h"

#include <algorithm>
#include <cstring>
#include <istream>
#include <vector>

namespace Hash {

namespace {

// floor(frac(cbrt(p)) * 2^32) for the first 64 primes.
constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::uint32_t RotateRight(const std::uint32_t value, const unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
}

}  // namespace

void Sha256::compress(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t i = 0; i < 16; ++i) {
        schedule[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24)
                    | (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16)
                    | (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8)
                    |  static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t previous = schedule[i - 15];
        const std::uint32_t recent = schedule[i - 2];
        const std::uint32_t s0 = RotateRight(previous, 7) ^ RotateRight(previous, 18) ^ (previous >> 3);
        const std::uint32_t s1 = RotateRight(recent, 17) ^ RotateRight(recent, 19) ^ (recent >> 10);
        schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t sigma1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
        const std::uint32_t choice = (e & f) ^ (~e & g);
        const std::uint32_t temp1 = h + sigma1 + choice + kRoundConstants[i] + schedule[i];
        const std::uint32_t sigma0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sigma0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    totalBytes += size;

    if (buffered > 0) {
        const std::size_t take = std::min(size, buffer.size() - buffered);
        std::memcpy(buffer.data() + buffered, bytes, take);
        buffered += take;
        bytes += take;
        size -= take;
        if (buffered < buffer.size()) {
            return;
        }
        compress(buffer.data());
        buffered = 0;
    }

    while (size >= buffer.size()) {
        compress(bytes);
        bytes += buffer.size();
        size -= buffer.size();
    }

    std::memcpy(buffer.data(), bytes, size);
    buffered = size;
}

Sha256Digest Sha256::finish() {
    // Taken before the padding, which goes through update() and would count.
    const std::uint64_t bitLength = totalBytes * 8;

    constexpr std::uint8_t one = 0x80;
    constexpr std::uint8_t zero = 0;
    update(&one, 1);
    while (buffered != buffer.size() - sizeof(std::uint64_t)) {
        update(&zero, 1);
    }

    std::array<std::uint8_t, sizeof(std::uint64_t)> length{};
    for (std::size_t i = 0; i < length.size(); ++i) {
        length[i] = static_cast<std::uint8_t>(bitLength >> (56 - i * 8));
    }
    update(length.data(), length.size());

    Sha256Digest digest{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        digest[i * 4]     = static_cast<std::uint8_t>(state[i] >> 24);
        digest[i * 4 + 1] = static_cast<std::uint8_t>(state[i] >> 16);
        digest[i * 4 + 2] = static_cast<std::uint8_t>(state[i] >> 8);
        digest[i * 4 + 3] = static_cast<std::uint8_t>(state[i]);
    }
    return digest;
}

Sha256Digest Sha256Of(const void* data, const std::size_t size) {
    Sha256 hash;
    hash.update(data, size);
    return hash.finish();
}

Sha256Digest Sha256OfStream(std::istream& in) {
    Sha256 hash;
    std::vector<char> chunk(64 * 1024);
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        // A short read is how the last chunk arrives, not a failure.
        const auto read = in.gcount();
        if (read <= 0) {
            break;
        }
        hash.update(chunk.data(), static_cast<std::size_t>(read));
    }
    return hash.finish();
}

std::string ToHex(const Sha256Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(digest.size() * 2);
    for (const auto byte : digest) {
        hex.push_back(digits[byte >> 4]);
        hex.push_back(digits[byte & 0x0F]);
    }
    return hex;
}

}  // namespace Hash
