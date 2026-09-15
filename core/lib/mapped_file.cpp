#include "core/lib/mapped_file.h"

#include "core/lib/file_stream.h"

#include <istream>
#include <limits>
#include <string>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#define MATRIXGUI_HAS_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define MATRIXGUI_HAS_MMAP 0
#endif

namespace Io {

#if MATRIXGUI_HAS_MMAP

MappedFile::MappedFile(const std::filesystem::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw Error("Can't open file for mapping: " + path.string());
    }

    struct stat info {};
    if (::fstat(descriptor, &info) != 0) {
        ::close(descriptor);
        throw Error("Can't stat file for mapping: " + path.string());
    }
    length = static_cast<std::size_t>(info.st_size);

    if (length > 0) {
        void* mapped = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (mapped == MAP_FAILED) {
            ::close(descriptor);
            length = 0;
            throw Error("Can't map file: " + path.string());
        }
        address = mapped;
    }
    ::close(descriptor);
}

bool MappedFile::IsSupported() {
    return true;
}

void MappedFile::adviseWillNeed() {
    if (address != nullptr) {
        ::madvise(address, length, MADV_WILLNEED);
    }
}

void MappedFile::adviseSequential() {
    if (address != nullptr) {
        ::madvise(address, length, MADV_SEQUENTIAL);
    }
}

void MappedFile::unmap() {
    if (address != nullptr) {
        ::munmap(address, length);
    }
    address = nullptr;
    length = 0;
}

#else

MappedFile::MappedFile(const std::filesystem::path& path) {
    throw Error("File mapping is not supported on this platform: " + path.string());
}

bool MappedFile::IsSupported() {
    return false;
}

void MappedFile::adviseWillNeed() { }

void MappedFile::adviseSequential() { }

void MappedFile::unmap() {
    address = nullptr;
    length = 0;
}

#endif

MappedFile::~MappedFile() {
    unmap();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : address{std::exchange(other.address, nullptr)}
    , length{std::exchange(other.length, 0)}
{ }

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        unmap();
        address = std::exchange(other.address, nullptr);
        length = std::exchange(other.length, 0);
    }
    return *this;
}

std::size_t AvailableMemoryBytes() {
    const auto available = TryReadFile("/proc/meminfo", [](std::istream& in) -> std::size_t {
        std::string key;
        while (in >> key) {
            if (key == "MemAvailable:") {
                std::size_t kilobytes = 0;
                if (in >> kilobytes) {
                    return kilobytes * 1024;
                }
                return 0;
            }
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0;
    });
    return available.value_or(0);
}

}  // namespace Io
