#include <doctest/doctest.h>

#include "core/lib/cache.h"

#include <string>

namespace {

using Cache = cache::LRUCache<std::string, double>;

Cache Sized(const std::size_t size) {
    return Cache{cache::CacheConfig<std::string, double>{.max_size = size}};
}

}  // namespace


TEST_CASE("A cache returns what was put in it, and nullopt otherwise") {
    auto cache = Sized(4);
    CHECK_FALSE(cache.get("x").has_value());

    cache.put("x", 1.5);
    REQUIRE(cache.get("x").has_value());
    CHECK(*cache.get("x") == doctest::Approx(1.5));
    CHECK(cache.contains("x"));
    CHECK(cache.size() == 1);
}

TEST_CASE("Putting an existing key overwrites the value without growing the cache") {
    auto cache = Sized(4);
    cache.put("x", 1.0);
    cache.put("x", 2.0);
    CHECK(cache.size() == 1);
    CHECK(*cache.get("x") == doctest::Approx(2.0));
}

TEST_CASE("The cache evicts the least recently used entry at capacity") {
    auto cache = Sized(2);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.put("c", 3);

    CHECK(cache.size() == 2);
    CHECK_FALSE(cache.get("a").has_value());  // oldest, dropped
    CHECK(cache.get("b").has_value());
    CHECK(cache.get("c").has_value());
}

TEST_CASE("A read counts as use, so it protects an entry from eviction") {
    // This is what makes the rank cache worth anything: the organisms that keep
    // being looked up are the ones that stay.
    auto cache = Sized(2);
    cache.put("a", 1);
    cache.put("b", 2);
    (void)cache.get("a");  // "a" is now the most recent
    cache.put("c", 3);

    CHECK(cache.get("a").has_value());
    CHECK_FALSE(cache.get("b").has_value());
}

TEST_CASE("Re-putting an existing key also refreshes its position") {
    auto cache = Sized(2);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.put("a", 10);
    cache.put("c", 3);

    CHECK(*cache.get("a") == doctest::Approx(10));
    CHECK_FALSE(cache.get("b").has_value());
}

TEST_CASE("clear empties the cache") {
    auto cache = Sized(4);
    cache.put("a", 1);
    cache.put("b", 2);
    cache.clear();

    CHECK(cache.empty());
    CHECK(cache.size() == 0);
    CHECK_FALSE(cache.get("a").has_value());
}

TEST_CASE("remove drops one entry and reports whether there was one") {
    auto cache = Sized(4);
    cache.put("a", 1);
    CHECK(cache.remove("a"));
    CHECK_FALSE(cache.remove("a"));
    CHECK(cache.empty());
}

TEST_CASE("The default capacity is a thousand entries") {
    // The rank cache is default-constructed, so this number is the one it runs
    // with against a world that is two orders of magnitude larger.
    const cache::CacheConfig<std::string, double> defaults;
    CHECK(defaults.max_size == 1000);
}

TEST_CASE("Statistics are off by default and count hits and misses when on") {
    auto counting = Cache{cache::CacheConfig<std::string, double>{
        .max_size = 2, .enable_stats = true}};
    counting.put("a", 1);
    (void)counting.get("a");
    (void)counting.get("b");

    const auto stats = counting.get_stats();
    CHECK(stats.hits == 1);
    CHECK(stats.misses == 1);
    CHECK(stats.hit_ratio() == doctest::Approx(0.5));

    auto silent = Sized(2);
    silent.put("a", 1);
    (void)silent.get("a");
    CHECK(silent.get_stats().hits == 0);
}
