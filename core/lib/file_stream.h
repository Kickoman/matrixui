#pragma once

#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace Io {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline std::ifstream OpenForReading(const std::filesystem::path& path, const std::ios::openmode mode = {}) {
    std::ifstream file(path, std::ios::in | mode);
    if (!file) {
        throw Error("Can't open file for reading: " + path.string());
    }
    return file;
}

inline std::ofstream OpenForWriting(const std::filesystem::path& path, const std::ios::openmode mode = {}) {
    std::ofstream file(path, std::ios::out | mode);
    if (!file) {
        throw Error("Can't open file for writing: " + path.string());
    }
    return file;
}

inline std::optional<std::ifstream> TryOpenForReading(
    const std::filesystem::path& path,
    const std::ios::openmode mode = {}
) {
    std::ifstream file(path, std::ios::in | mode);
    if (!file) {
        return std::nullopt;
    }
    return file;
}

template <typename F>
decltype(auto) ReadFile(const std::filesystem::path& path, F&& read, const std::ios::openmode mode = {}) {
    auto file = OpenForReading(path, mode);
    try {
        return std::forward<F>(read)(file);
    } catch (const std::exception& error) {
        throw Error(std::string(error.what()) + ": " + path.string());
    }
}

template <typename F>
auto TryReadFile(const std::filesystem::path& path, F&& read, const std::ios::openmode mode = {})
    -> std::optional<decltype(read(std::declval<std::istream&>()))> {
    auto file = TryOpenForReading(path, mode);
    if (!file) {
        return std::nullopt;
    }
    try {
        return std::forward<F>(read)(*file);
    } catch (const std::exception& error) {
        throw Error(std::string(error.what()) + ": " + path.string());
    }
}

template <typename F>
void WriteFile(const std::filesystem::path& path, F&& write, const std::ios::openmode mode = {}) {
    auto file = OpenForWriting(path, mode);
    try {
        std::forward<F>(write)(file);
    } catch (const std::exception& error) {
        throw Error(std::string(error.what()) + ": " + path.string());
    }
}

}  // namespace Io
