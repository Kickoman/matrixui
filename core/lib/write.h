#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

constexpr bool IsLittleEndian() {
    return std::endian::native == std::endian::little;
}

inline std::uint16_t SwapBytes(std::uint16_t val) {
    return (val >> 8) | (val << 8);
}

inline std::uint32_t SwapBytes(std::uint32_t val) {
    return ((val >> 24) & 0xFF) |
           ((val >> 8) & 0xFF00) |
           ((val << 8) & 0xFF0000) |
           ((val << 24) & 0xFF000000);
}

inline std::uint64_t SwapBytes(std::uint64_t val) {
    return ((val & 0x00000000000000FFULL) << 56) |
           ((val & 0x000000000000FF00ULL) << 40) |
           ((val & 0x0000000000FF0000ULL) << 24) |
           ((val & 0x00000000FF000000ULL) << 8)  |
           ((val & 0x000000FF00000000ULL) >> 8)  |
           ((val & 0x0000FF0000000000ULL) >> 24) |
           ((val & 0x00FF000000000000ULL) >> 40) |
           ((val & 0xFF00000000000000ULL) >> 56);
}

template<typename T>
void WriteBinaryLE(std::ofstream& file, T value) {
    if constexpr (!IsLittleEndian()) {
        if constexpr (sizeof(T) == 2) {
            value = SwapBytes(static_cast<std::uint16_t>(value));
        } else if constexpr (sizeof(T) == 4) {
            value = SwapBytes(static_cast<std::uint32_t>(value));
        } else if constexpr (sizeof(T) == 8) {
            value = SwapBytes(static_cast<std::uint64_t>(value));
        }
    }
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
void ReadBinaryLE(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(T));

    if constexpr (!IsLittleEndian()) {
        if constexpr (sizeof(T) == 2) {
            value = SwapBytes(static_cast<std::uint16_t>(value));
        } else if constexpr (sizeof(T) == 4) {
            value = SwapBytes(static_cast<std::uint32_t>(value));
        } else if constexpr (sizeof(T) == 8) {
            value = SwapBytes(static_cast<std::uint64_t>(value));
        }
    }
}

template<>
inline void WriteBinaryLE(std::ofstream& file, double value) {
    std::uint64_t temp;
    std::memcpy(&temp, &value, sizeof(double));
    WriteBinaryLE(file, temp);
}

template<>
inline void ReadBinaryLE(std::ifstream& file, double& value) {
    std::uint64_t temp;
    ReadBinaryLE(file, temp);
    std::memcpy(&value, &temp, sizeof(double));
}

template<typename T>
inline void WriteBulkLE(std::ofstream& file, const std::vector<T>& values) {
    if constexpr (IsLittleEndian()) {
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    } else {
        for (T val : values) {
            WriteBinaryLE(file, val);
        }
    }
}

template<typename T>
inline void ReadBulkLE(std::ifstream& file, std::vector<T>& values) {
    if constexpr (IsLittleEndian()) {
        file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    } else {
        for (T& val : values) {
            ReadBinaryLE(file, val);
        }
    }
}
