#pragma once

#include <concepts>
#include <type_traits>
#include <tuple>
#include <functional>
#include <cstddef>
#include <utility>

namespace ArbitraryCache {

template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    seed ^= std::hash<T>{}(value)
          + 0x9e3779b97f4a7c15ULL
          + (seed << 6) + (seed >> 2);
}

struct TupleHash {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& tuple) const {
        std::size_t seed = 0;
        std::apply(
            [&seed](const auto&... elems) {
                (hash_combine(seed, elems), ...);
            },
            tuple);
        return seed;
    }
};

template <typename O, typename Value>
concept OptionalLike = requires(const O& o) {
    { o.has_value() } -> std::convertible_to<bool>;
    { *o }            -> std::convertible_to<Value>;
};

template <typename C, typename Key, typename Value>
concept CacheFor = requires(C& cache, const Key& key, Value value) {
    { cache.get(key) } -> OptionalLike<Value>;
    cache.put(key, value);
};

template<typename Cache, typename Func, typename... Args>
requires std::invocable<Func, Args...>
          && (std::copy_constructible<std::decay_t<Args>> && ...)
          && (!std::is_void_v<std::invoke_result_t<Func, Args...>>)
          && CacheFor<Cache,
                      std::tuple<std::decay_t<Args>...>,
                      std::invoke_result_t<Func, Args...>>
auto DoCached(Cache& cache, Func&& func, Args&&... args) -> std::invoke_result_t<Func, Args...> {
    auto key = std::make_tuple(args...);
    if (auto cached = cache.get(key); cached.has_value()) {
        return *cached;
    }

    auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    cache.put(std::move(key), result);
    return result;
}

}
