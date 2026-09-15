#pragma once

#include <cstddef>
#include <filesystem>

namespace Io {

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    static bool IsSupported();

    const std::byte* getData() const { return static_cast<const std::byte*>(address); }
    std::size_t getSize() const { return length; }

    void adviseWillNeed();
    void adviseSequential();

private:
    void unmap();

    void* address{nullptr};
    std::size_t length{0};
};

std::size_t AvailableMemoryBytes();

}  // namespace Io
