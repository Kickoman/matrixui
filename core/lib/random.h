#pragma once


#include <cstdint>


class XorShift {
public:
    explicit XorShift(std::uint64_t seed = 88172645463325252ULL)
        : state{seed ? seed : 1}
    { }

    std::uint64_t nextInteger() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    /* uniform [0, 1) */
    double nextDouble() {
        return static_cast<double>(nextInteger() >> 11) * 0x1.0p-53;
    }

    /* uniform [0, bound) */
    std::uint64_t nextInteger(std::uint64_t bound) {
        return nextInteger() % bound;
    }

private:
    std::uint64_t state;
};
