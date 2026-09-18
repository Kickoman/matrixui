#pragma once

#include <cstdint>

namespace Words {

using TWordId = std::uint32_t;
using TBucketId = std::uint32_t;

struct Pair {
    TWordId center;
    TWordId context;
};

}
