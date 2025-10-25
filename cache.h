#pragma once

#include <unordered_map>
#include <list>
#include <optional>
#include <functional>
#include <string>
#include <type_traits>

namespace cache {

struct CacheStats {
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t evictions = 0;
    std::size_t insertions = 0;

    double hit_ratio() const {
        const auto total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    }
};

template<typename Key, typename Value>
struct CacheConfig {
    std::size_t max_size = 1000;
    std::size_t max_memory_bytes = 0;
    std::function<std::size_t(const Key&, const Value&)> memory_calculator = nullptr;
    bool enable_stats = false;
};

// template<typename Key, typename Value>
// std::size_t default_memory_calculator(const Key& key, const Value& value) {
//     std::size_t size = 0;

//     if constexpr (std::is_arithmetic_v<Key>) {
//         size += sizeof(Key);
//     } else if constexpr (std::is_same_v<Key, std::string>) {
//         size += key.capacity();
//     } else if constexpr (std::is_same_v<Key, std::vector<typename Key::value_type>>) {
//         size += key.capacity() * sizeof(typename Key::value_type);
//     } else {
//         size += sizeof(Key);
//     }

//     if constexpr (std::is_arithmetic_v<Value>) {
//         size += sizeof(Value);
//     } else if constexpr (std::is_same_v<Value, std::string>) {
//         size += value.capacity();
//     } else if constexpr (std::is_same_v<Value, std::vector<typename Value::value_type>>) {
//         size += value.capacity() * sizeof(typename Value::value_type);
//     } else {
//         size += sizeof(Value);
//     }

//     return size;
// }

template<typename Key, typename Value>
std::size_t default_memory_calculator(const Key& key, const Value& value) {
    std::size_t size = 0;

    if constexpr (std::is_arithmetic_v<Key>) {
        size += sizeof(Key);
    } else if constexpr (std::is_same_v<Key, std::string>) {
        size += key.capacity();
    } else {
        size += sizeof(Key);
    }

    if constexpr (std::is_arithmetic_v<Value>) {
        size += sizeof(Value);
    } else if constexpr (std::is_same_v<Value, std::string>) {
        size += value.capacity();
    } else {
        size += sizeof(Value);
    }

    return size;
}

template<typename Key, typename Value>
class LRUCache {
private:
    using ListIterator = typename std::list<std::pair<Key, Value>>::iterator;

    struct CacheEntry {
        Value value;
        ListIterator list_iterator;
        std::size_t memory_usage;

        CacheEntry(Value&& val, ListIterator iter, std::size_t mem)
            : value(std::move(val)), list_iterator(iter), memory_usage(mem) {}

        CacheEntry(CacheEntry&& other) noexcept
            : value(std::move(other.value))
            , list_iterator(other.list_iterator)
            , memory_usage(other.memory_usage) {}

        CacheEntry& operator=(CacheEntry&& other) noexcept {
            if (this != &other) {
                value = std::move(other.value);
                list_iterator = other.list_iterator;
                memory_usage = other.memory_usage;
            }
            return *this;
        }

        CacheEntry(const CacheEntry&) = delete;
        CacheEntry& operator=(const CacheEntry&) = delete;
    };

public:
    explicit LRUCache(const CacheConfig<Key, Value>& config = {})
        : config_(config) {

        if (config_.max_memory_bytes > 0 && !config_.memory_calculator) {
            config_.memory_calculator = default_memory_calculator<Key, Value>;
        }

        if (config_.max_size > 0) {
            cache_map_.reserve(config_.max_size);
        }
    }

    LRUCache(const LRUCache&) = delete;
    LRUCache& operator=(const LRUCache&) = delete;

    LRUCache(LRUCache&&) = default;
    LRUCache& operator=(LRUCache&&) = default;

    void put(const Key& key, Value value) {
        auto it = cache_map_.find(key);

        if (it != cache_map_.end()) {
            auto& entry = it->second;

            if (config_.max_memory_bytes > 0) {
                const auto old_memory = entry.memory_usage;
                const auto new_memory = calculate_memory_usage(key, value);
                current_memory_usage_ += (new_memory - old_memory);
                entry.memory_usage = new_memory;
            }

            entry.value = std::move(value);
            cache_list_.splice(cache_list_.begin(), cache_list_, entry.list_iterator);
            entry.list_iterator->second = entry.value; // Update value in list
        } else {
            const auto memory_usage = config_.max_memory_bytes > 0 ?
                calculate_memory_usage(key, value) : 0;

            while ((config_.max_size > 0 && cache_map_.size() >= config_.max_size) ||
                   (config_.max_memory_bytes > 0 &&
                    current_memory_usage_ + memory_usage > config_.max_memory_bytes)) {
                evict_lru();
            }

            cache_list_.emplace_front(key, value);
            auto list_it = cache_list_.begin();

            CacheEntry entry{std::move(value), list_it, memory_usage};
            auto [map_it, inserted] = cache_map_.try_emplace(key, std::move(entry));

            if (inserted) {
                current_memory_usage_ += memory_usage;
                if (config_.enable_stats) {
                    stats_.insertions++;
                }
            }
        }
    }

    std::optional<Value> get(const Key& key) {
        auto it = cache_map_.find(key);

        if (it == cache_map_.end()) {
            if (config_.enable_stats) {
                stats_.misses++;
            }
            return std::nullopt;
        }

        auto& entry = it->second;
        cache_list_.splice(cache_list_.begin(), cache_list_, entry.list_iterator);

        if (config_.enable_stats) {
            stats_.hits++;
        }

        return entry.value;
    }

    bool remove(const Key& key) {
        auto it = cache_map_.find(key);

        if (it == cache_map_.end()) {
            return false;
        }

        const auto& entry = it->second;

        if (config_.max_memory_bytes > 0) {
            current_memory_usage_ -= entry.memory_usage;
        }

        cache_list_.erase(entry.list_iterator);
        cache_map_.erase(it);

        return true;
    }

    void clear() {
        cache_map_.clear();
        cache_list_.clear();
        current_memory_usage_ = 0;
    }

    bool contains(const Key& key) const {
        return cache_map_.find(key) != cache_map_.end();
    }

    std::size_t size() const { return cache_map_.size(); }
    std::size_t memory_usage() const { return current_memory_usage_; }
    bool empty() const { return cache_map_.empty(); }

    CacheStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = CacheStats{}; }

    auto begin() { return cache_list_.begin(); }
    auto end() { return cache_list_.end(); }
    auto begin() const { return cache_list_.begin(); }
    auto end() const { return cache_list_.end(); }

    const CacheConfig<Key, Value>& get_config() const { return config_; }

    const auto& get_list() const { return cache_list_; }
    const auto& get_map() const { return cache_map_; }

private:
    std::unordered_map<Key, CacheEntry> cache_map_;
    std::list<std::pair<Key, Value>> cache_list_;
    CacheConfig<Key, Value> config_;
    CacheStats stats_;
    std::size_t current_memory_usage_ = 0;

    void evict_lru() {
        if (cache_list_.empty()) return;

        const auto& lru_item = cache_list_.back();
        const auto key = lru_item.first;

        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            if (config_.max_memory_bytes > 0) {
                current_memory_usage_ -= it->second.memory_usage;
            }

            cache_map_.erase(it);
            cache_list_.pop_back();

            if (config_.enable_stats) {
                stats_.evictions++;
            }
        }
    }

    std::size_t calculate_memory_usage(const Key& key, const Value& value) const {
        if (config_.memory_calculator) {
            return config_.memory_calculator(key, value);
        }
        return 0;
    }
};

using IntCache = LRUCache<int, int>;
using StringCache = LRUCache<std::string, std::string>;
using StringIntCache = LRUCache<std::string, int>;
using IntStringCache = LRUCache<int, std::string>;

}
