#pragma once

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include <unistd.h>

namespace Tests {

// A scratch directory removed on destruction. Tests that need real files
// (every Save/Load round-trip in the words module goes through the filesystem)
// build them here instead of polluting the working directory.
class TempDir {
public:
    TempDir() {
        static std::atomic<unsigned> counter{0};
        const auto unique = std::to_string(::getpid()) + "_" + std::to_string(counter++);
        path = std::filesystem::temp_directory_path() / ("matrixgui_test_" + unique);
        std::filesystem::create_directories(path);
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::filesystem::path file(const std::string& name) const { return path / name; }

    // Writes `content` into `name` and returns the path.
    std::filesystem::path write(const std::string& name, const std::string& content) const {
        const auto target = file(name);
        std::ofstream out(target, std::ios::binary);
        out << content;
        return target;
    }

private:
    std::filesystem::path path;
};

}  // namespace Tests
